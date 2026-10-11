"""Domain models for the role-play loop — the data contracts the agents pass around.

`ScenarioConfig` is the "product pack" input (persona + objective + rubric). `Scorecard`
is the Evaluator's structured output — it doubles as the JSON schema handed to the model's
structured-output API (Groq `response_format`/Gemini `response_schema`), so keep it flat and
unambiguous: providers reject or degrade on deeply-nested / overly-constrained schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioConfig(BaseModel):
    """One role-play scene. This is the swappable "pack" — change these fields (not code)
    to go from a job-interview coach to a sales-objection trainer."""

    title: str
    persona: str  # who the Role-Player IS ("a skeptical hiring manager for a backend role")
    objective: str  # what the LEARNER is trying to do ("pass a behavioral interview")
    difficulty: str = "medium"  # easy | medium | hard — nudges curveball intensity
    curveballs: list[str] = Field(
        default_factory=list,
        description="Objections/escalations the Scenario-Director can inject, one per turn.",
    )
    rubric: list[str] = Field(
        default_factory=list,
        description="Criteria names the Evaluator scores (e.g. 'clarity', 'STAR structure').",
    )


class CriterionScore(BaseModel):
    """One rubric line in the scorecard. `evidence` MUST be a verbatim quote from the
    transcript — moment-level evidence is the 2026 differentiator (not one vague number)."""

    criterion: str
    score: int = Field(ge=1, le=5)
    evidence: str  # a direct quote from the learner's turns
    rationale: str  # one sentence: why this score


class Scorecard(BaseModel):
    """The Evaluator's full output for a session."""

    criteria: list[CriterionScore]
    overall: int = Field(ge=1, le=5)
    summary: str
    focus_next: str  # the single thing to work on next retry


# A ready-to-run sample so the loop is demoable from turn one (curl / tests / UI).
SAMPLE_SCENARIO = ScenarioConfig(
    title="Behavioral interview — 'tell me about a conflict'",
    persona="a friendly but probing engineering manager interviewing for a senior backend role",
    objective="answer behavioral questions using the STAR structure, concisely and with impact",
    difficulty="medium",
    curveballs=[
        "Open by asking them to tell you about themselves.",
        "Now ask them to describe a conflict they faced and how they handled it.",
        "Push them to quantify the impact of that with a real number.",
    ],
    rubric=["clarity", "STAR structure", "specificity/evidence", "confidence"],
)
