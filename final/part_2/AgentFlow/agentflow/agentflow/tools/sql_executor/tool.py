import os
import sqlite3
from agentflow.tools.base import BaseTool

TOOL_NAME = "SQL_Executor_Tool"

LIMITATIONS = """
1. Only supports SQLite databases.
2. Requires BIRD_DB_DIR environment variable to be set to the directory containing database folders.
3. Read-only queries only; does not support INSERT, UPDATE, DELETE.
"""

BEST_PRACTICES = """
1. Use this tool to execute a SQL query against a specific BIRD database.
2. Provide the db_id (database name) and the SQL query to execute.
3. The tool returns the query results as a formatted string.
4. Use it to verify SQL queries generated for the BIRD Text-to-SQL benchmark.
5. If unsure about the schema, query sqlite_master first: SELECT sql FROM sqlite_master WHERE type='table';
"""


class SQL_Executor_Tool(BaseTool):
    def __init__(self, model_string=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="Executes a SQL query against a BIRD benchmark SQLite database and returns the results.",
            tool_version="1.0.0",
            input_types={
                "sql_query": "str - The SQL query to execute.",
                "db_id": "str - The database identifier (folder name under BIRD_DB_DIR).",
            },
            output_type="str - Query results as a formatted string, or an error message.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(sql_query="SELECT COUNT(*) FROM singer", db_id="concert_singer")',
                    "description": "Count rows in the singer table of the concert_singer database."
                },
            ],
            user_metadata={
                "limitation": LIMITATIONS,
                "best_practice": BEST_PRACTICES,
            },
        )

    def execute(self, sql_query: str = None, db_id: str = None, query: str = None, **kwargs) -> str:
        # Accept 'query' as alias for 'sql_query'
        if sql_query is None and query is not None:
            sql_query = query
        if not sql_query:
            return "Error: sql_query is required."

        bird_db_dir = os.environ.get("BIRD_DB_DIR", "")
        if not bird_db_dir:
            return "Error: BIRD_DB_DIR environment variable is not set."

        # If db_id not provided, list available databases
        if not db_id:
            try:
                dbs = [d for d in os.listdir(bird_db_dir) if os.path.isdir(os.path.join(bird_db_dir, d))]
                return f"Error: db_id is required. Available databases: {', '.join(sorted(dbs))}"
            except Exception:
                return "Error: db_id is required."

        db_path = os.path.join(bird_db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return f"Error: Database not found at {db_path}"

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "Query executed successfully. No results returned."

            col_names = rows[0].keys()
            header = " | ".join(col_names)
            separator = "-" * len(header)
            result_lines = [header, separator]
            for row in rows[:50]:  # cap at 50 rows
                result_lines.append(" | ".join(str(v) for v in row))

            if len(rows) > 50:
                result_lines.append(f"... ({len(rows)} rows total, showing first 50)")

            return "\n".join(result_lines)

        except Exception as e:
            return f"SQL execution error: {str(e)}"
