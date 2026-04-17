import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from agentflow.tools.base import BaseTool

TOOL_NAME = "Ground_Serper_Search_Tool"

LIMITATIONS = """
1. This tool is only suitable for general information search via Google Search results.
2. Results are snippets from web pages, not full page content.
3. Not suitable for searching and analyzing videos at YouTube or other video platforms.
"""

BEST_PRACTICES = """
1. Choose this tool when you want to search general information about a topic.
2. Best for factual questions such as "What is the capital of France?" or "Who won the 2024 UEFA Euro?"
3. The tool returns titles, URLs, and text snippets from top Google results.
4. Ideal for world knowledge, definitions, and multi-hop question answering.
"""


class Serper_Search_Tool(BaseTool):
    def __init__(self, model_string=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A web search tool powered by Serper.dev that retrieves real-time Google Search results including titles, URLs, and snippets.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "num_results": "int - Number of search results to return. Default is 10.",
            },
            output_type="str - Formatted search results containing titles, URLs, and snippets.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the capital of France?")',
                    "description": "Search for general information about the capital of France."
                },
                {
                    "command": 'execution = tool.execute(query="Who won the UEFA Euro 2024?", num_results=5)',
                    "description": "Search with a limited number of results."
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            }
        )
        self.max_retries = 3
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise Exception("Serper API key not found. Please set the SERPER_API_KEY environment variable.")
        self.endpoint = "https://google.serper.dev/search"

    def _execute_search(self, query: str, num_results: int) -> str:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=15,
                )
                response.raise_for_status()
                data = response.json()
                return self._format_results(data)
            except Exception as e:
                print(f"Serper search attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return f"Serper search failed after {self.max_retries} attempts. Last error: {e}"

        return "Serper search failed to get a valid response."

    def _format_results(self, data: dict) -> str:
        parts = []

        # Answer box (quick answer)
        if "answerBox" in data:
            box = data["answerBox"]
            answer = box.get("answer") or box.get("snippet") or ""
            if answer:
                parts.append(f"Quick Answer: {answer}\n")

        # Knowledge graph
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            desc = kg.get("description", "")
            if desc:
                parts.append(f"Knowledge Graph: {desc}\n")

        # Organic results
        organic = data.get("organic", [])
        for i, result in enumerate(organic, 1):
            title = result.get("title", "")
            link = result.get("link", "")
            snippet = result.get("snippet", "")
            parts.append(f"[{i}] {title}\nURL: {link}\nSnippet: {snippet}\n")

        if not parts:
            return "No results found."

        return "\n".join(parts)

    def execute(self, query: str, num_results: int = 10) -> str:
        """
        Execute the Serper search tool.

        Parameters:
            query (str): The search query.
            num_results (int): Number of results to return. Default is 10.

        Returns:
            str: Formatted search results.
        """
        return self._execute_search(query, num_results)

    def get_metadata(self):
        return super().get_metadata()


if __name__ == "__main__":
    tool = Serper_Search_Tool()

    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(json.dumps(metadata, indent=4))

    examples = [
        {"query": "What is the capital of France?"},
        {"query": "Who won the UEFA Euro 2024?", "num_results": 5},
    ]

    for ex in examples:
        print(f"\nSearching: {ex['query']}")
        result = tool.execute(**ex)
        print(result)
        print("-" * 50)
