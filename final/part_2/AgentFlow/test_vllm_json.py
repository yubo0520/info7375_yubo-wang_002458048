"""
Quick test: verify that ChatVLLM returns proper Pydantic objects
when response_format is specified (structured JSON output).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentflow.agentflow.engine.vllm import ChatVLLM
from agentflow.agentflow.models.formatters import NextStep, QueryAnalysis

VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://10.253.196.237:8000/v1")
MODEL    = os.environ.get("VLLM_MODEL",    "lovedheart/Qwen3.5-9B-FP8")

engine = ChatVLLM(model_string=MODEL, base_url=VLLM_URL, use_cache=False)
print(f"Connected to {VLLM_URL}  model={MODEL}\n")

# ── Test 1: plain text ────────────────────────────────────────────────────────
print("=" * 60)
print("Test 1: plain text (no response_format)")
result = engine("Say hello in one sentence.")
assert isinstance(result, str), f"Expected str, got {type(result)}"
print(f"  type : {type(result)}")
print(f"  value: {result[:120]}")
print("  PASS\n")

# ── Test 2: QueryAnalysis ─────────────────────────────────────────────────────
print("=" * 60)
print("Test 2: structured output -> QueryAnalysis")
result = engine(
    "Analyse the query: 'Which film director is older, the director of Titanic or the director of Inception?'",
    response_format=QueryAnalysis,
)
print(f"  type : {type(result)}")
assert isinstance(result, QueryAnalysis), f"Expected QueryAnalysis, got {type(result)}"
print(f"  concise_summary       : {result.concise_summary[:80]}")
print(f"  required_skills       : {result.required_skills[:80]}")
print(f"  relevant_tools        : {result.relevant_tools[:80]}")
print(f"  additional_considerations: {result.additional_considerations[:80]}")
print("  PASS\n")

# ── Test 3: NextStep ──────────────────────────────────────────────────────────
print("=" * 60)
print("Test 3: structured output -> NextStep")
result = engine(
    (
        "Given the query 'Which film director is older, Titanic or Inception director?', "
        "choose the next tool to use from: ['Wikipedia_Search_Tool', 'Base_Generator_Tool']. "
        "Return context, sub_goal, tool_name, and justification."
    ),
    response_format=NextStep,
)
print(f"  type      : {type(result)}")
assert isinstance(result, NextStep), f"Expected NextStep, got {type(result)}"
print(f"  tool_name : {result.tool_name}")
print(f"  sub_goal  : {result.sub_goal[:80]}")
print(f"  context   : {result.context[:80]}")
print("  PASS\n")

print("=" * 60)
print("All tests passed.")
