"""Phase 1 — multi-agent text core.

A LangGraph graph orchestrating three agents over a shared state:
  - Scenario-Director : sets the scene + injects curveballs (rule-based in Phase 1)
  - Role-Player       : *is* the persona, improvises in character   [your TODO]
  - Evaluator         : LLM-as-judge → rubric scorecard (structured JSON)  [your TODO]

See `graph.py` for the wiring and `schemas.py` for the domain models.
"""
