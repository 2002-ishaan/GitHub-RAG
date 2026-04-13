"""
evaluation/stress_test.py
────────────────────────────────────────────────────────────────
Broader robustness stress test for GitHub docs + action intents.

Covers:
- easy / medium / hard / vague GitHub documentation questions
- spelling mistakes and natural-language variants
- billing and ticketing action routes

Run:
    python -m evaluation.stress_test

Output:
    evaluation/stress_results.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from configs.settings import load_settings, setup_logging
from generation.rag_chain import RAGChain
from agent.intent_router import IntentRouter


@dataclass
class StressCase:
    question: str
    expected_intent: str
    difficulty: str
    category: str


def build_cases() -> list[StressCase]:
    easy = [
        "What is GitHub?",
        "How do I create a repository on GitHub?",
        "How do I create a private repository?",
        "How do I fork a repository?",
        "How do I clone a repository?",
        "How do I create a pull request?",
        "How do I open an issue?",
        "How do I add collaborators?",
        "How do I set up 2FA?",
        "How do I generate a personal access token?",
        "How do I enable Dependabot alerts?",
        "How do I use GitHub Actions?",
        "How do I contact GitHub support?",
        "How do I cancel my GitHub subscription?",
        "What plans are available on GitHub?",
    ]

    medium = [
        "Can you walk me through making a repo private after creation?",
        "What are the steps for branch protection on main?",
        "How can I add team members in an organization?",
        "How do I configure CODEOWNERS for reviews?",
        "How do I use environments and secrets in Actions workflows?",
        "How can I turn on security scanning for my repository?",
        "What is the difference between fine-grained and classic PATs?",
        "How do I manage repository visibility in an org?",
        "How do I rotate SSH keys safely on GitHub?",
        "How can I request a billing refund through support?",
    ]

    hard = [
        "I need policy-level controls for push protection and secret scanning. What should I configure?",
        "How do rulesets differ from classic branch protection in GitHub?",
        "What should I check if code scanning alerts are not appearing?",
        "How do I design a safer release flow using protected branches and required reviews?",
        "How do I diagnose failing GitHub Actions that pass locally but fail in CI?",
        "What are best practices for organization-level access governance on GitHub?",
        "How do I secure dependency management using Dependabot across multiple repositories?",
        "What is the recommended process to harden authentication for admins?",
    ]

    vague = [
        "repo setup help",
        "how make it private",
        "cant login github 2fa",
        "billing issue with plan",
        "ticket status please",
        "need support now",
        "how do i do this in org",
        "pr not working",
        "branch rules confusion",
        "show me what i should do next",
    ]

    typo_variants = [
        "how do i create a reposotory on gihub",
        "how to open a tikcet",
        "check biling for alice",
        "how do i setup authentification",
        "how to add member to orgnization",
        "what is depedabot",
        "how do i make a pr in gihub",
        "how to cancel subscrption",
        "check tickt tkt-001",
        "how do i use repo secretes in actions",
    ]

    action_cases = [
        StressCase("create a support ticket", "create_ticket", "easy", "actions"),
        StressCase("open ticket for billing problem", "create_ticket", "easy", "actions"),
        StressCase("check ticket TKT-001", "check_ticket", "easy", "actions"),
        StressCase("check tickt tkt001", "check_ticket", "medium", "actions"),
        StressCase("check billing for alice", "check_billing", "easy", "actions"),
        StressCase("show my plan", "check_billing", "easy", "actions"),
        StressCase("register a new account for michael", "register_user", "medium", "actions"),
        StressCase("upgrade alice to enterprise", "upgrade_plan", "easy", "actions"),
        StressCase("list all accounts", "list_accounts", "easy", "actions"),
        StressCase("close ticket TKT-001", "close_ticket_by_id", "easy", "actions"),
        StressCase("close all tickets", "close_tickets", "easy", "actions"),
    ]

    cases: list[StressCase] = []

    for q in easy:
        cases.append(StressCase(q, "rag_query", "easy", "github_docs"))
    for q in medium:
        cases.append(StressCase(q, "rag_query", "medium", "github_docs"))
    for q in hard:
        cases.append(StressCase(q, "rag_query", "hard", "github_docs"))
    for q in vague:
        expected = "check_ticket" if "ticket" in q else "rag_query"
        cases.append(StressCase(q, expected, "vague", "mixed"))
    for q in typo_variants:
        if "biling" in q:
            expected = "check_billing"
        elif "open" in q and ("tikcet" in q or "tickt" in q):
            expected = "create_ticket"
        elif "tikcet" in q or "tickt" in q:
            expected = "check_ticket"
        else:
            expected = "rag_query"
        cases.append(StressCase(q, expected, "vague", "typo"))

    cases.extend(action_cases)
    return cases


def run() -> None:
    settings = load_settings()
    setup_logging(settings)

    chain = RAGChain(settings)
    router = IntentRouter(settings, chain.prompts)

    cases = build_cases()
    results = []

    intent_pass = 0
    rag_checked = 0
    rag_supported = 0
    total_latency = 0.0

    for idx, case in enumerate(cases, start=1):
        start = time.time()
        intent = router.classify(case.question).intent
        elapsed = time.time() - start
        total_latency += elapsed

        intent_ok = intent == case.expected_intent
        if intent_ok:
            intent_pass += 1

        rag_ok = None
        top_similarity = None

        if intent == "rag_query":
            rag_checked += 1
            rag_response = chain.ask(question=case.question)
            rag_ok = bool(rag_response.is_supported)
            if rag_ok:
                rag_supported += 1
            top_similarity = rag_response.metadata.get("top_similarity", 0.0)

        results.append(
            {
                "id": idx,
                "question": case.question,
                "difficulty": case.difficulty,
                "category": case.category,
                "expected_intent": case.expected_intent,
                "actual_intent": intent,
                "intent_pass": intent_ok,
                "rag_supported": rag_ok,
                "top_similarity": top_similarity,
                "latency_sec": round(elapsed, 3),
            }
        )

        print(
            f"[{idx:03d}] intent={intent} | expected={case.expected_intent} | "
            f"intent_pass={intent_ok} | q={case.question[:72]}"
        )

    summary = {
        "total_cases": len(cases),
        "intent_pass": intent_pass,
        "intent_accuracy": round(intent_pass / len(cases) * 100, 2),
        "rag_checked": rag_checked,
        "rag_supported": rag_supported,
        "rag_support_rate": round((rag_supported / rag_checked * 100) if rag_checked else 0.0, 2),
        "avg_intent_latency_sec": round(total_latency / len(cases), 3),
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    out_path = Path(__file__).parent / "stress_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== STRESS TEST SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    run()
