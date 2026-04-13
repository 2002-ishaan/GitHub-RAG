"""
evaluation/top200_questions.py
────────────────────────────────────────────────────────────────
Generate a professor-demo set of 200 GitHub documentation questions.

Why this exists:
- Demo questions are often paraphrased on the fly.
- Users make spelling mistakes.
- Retrieval quality should remain stable across wording changes.

Run:
    python -m evaluation.top200_questions

Output:
    evaluation/top200_questions.json
"""

from __future__ import annotations

import json
from pathlib import Path


BASE_QUESTIONS = [
    "How do I create a repository on GitHub?",
    "How do I create a private repository?",
    "How do I change repository visibility?",
    "How do I fork a repository?",
    "How do I clone a repository?",
    "How do I archive a repository?",
    "How do I delete a repository?",
    "How do I transfer a repository to another owner?",
    "How do I protect a branch?",
    "How do I create a pull request?",
    "How do I review a pull request?",
    "How do I merge a pull request?",
    "How do I resolve merge conflicts?",
    "How do I create an issue?",
    "How do I label issues?",
    "How do I create milestones?",
    "How do I use GitHub Projects?",
    "How do I add collaborators to a repository?",
    "How do I create a GitHub organization?",
    "How do I invite members to an organization?",
    "How do I manage team permissions?",
    "How do I set up CODEOWNERS?",
    "How do I configure branch rulesets?",
    "How do I create a release?",
    "How do I create tags on GitHub?",
    "How do I use GitHub Actions?",
    "How do I create an Actions workflow?",
    "How do I use workflow secrets?",
    "How do I cache dependencies in Actions?",
    "How do I use environments in Actions?",
    "How do I set up two-factor authentication?",
    "How do I reset two-factor authentication?",
    "How do I manage SSH keys?",
    "How do I create a personal access token?",
    "How do I manage fine-grained PAT permissions?",
    "How do I enable Dependabot alerts?",
    "How do I enable Dependabot security updates?",
    "How do I use secret scanning?",
    "How do I use code scanning?",
    "How do I enable private vulnerability reporting?",
    "What is GitHub?",
    "What is the difference between Git and GitHub?",
    "What is GitHub Copilot?",
    "How does GitHub billing work?",
    "What plans are available on GitHub?",
    "How do I upgrade my GitHub plan?",
    "How do I cancel my GitHub subscription?",
    "How do I request a GitHub refund?",
    "How do I manage payment methods on GitHub?",
    "How do I contact GitHub support?",
]


TYPO_MAP = {
    "github": "gihub",
    "repository": "reposotory",
    "authentication": "authentification",
    "organization": "orgnization",
    "permissions": "permissons",
    "billing": "biling",
    "subscription": "subscrption",
    "dependabot": "depedabot",
}


def _one_typo_variant(question: str) -> str:
    lower = question.lower()
    for good, bad in TYPO_MAP.items():
        if good in lower:
            return lower.replace(good, bad, 1)
    return lower.replace("how do i", "how i can", 1)


def _paraphrase_variant(question: str) -> str:
    q = question.strip().rstrip("?")
    if q.lower().startswith("how do i"):
        return q.replace("How do I", "Can you explain how to", 1) + "?"
    if q.lower().startswith("what is"):
        return q.replace("What is", "Could you explain", 1) + "?"
    return "Can you help with this: " + q + "?"


def _short_variant(question: str) -> str:
    q = question.lower()
    q = q.replace("how do i ", "")
    q = q.replace("what is ", "")
    q = q.replace("?", "")
    return q.strip().capitalize() + "?"


def build_top200() -> list[dict]:
    rows = []
    idx = 1
    for base in BASE_QUESTIONS:
        variants = [
            ("base", base),
            ("paraphrase", _paraphrase_variant(base)),
            ("typo", _one_typo_variant(base)),
            ("short", _short_variant(base)),
        ]
        for variant_type, question in variants:
            rows.append(
                {
                    "id": idx,
                    "base_question": base,
                    "variant_type": variant_type,
                    "question": question,
                }
            )
            idx += 1
    return rows


def main() -> None:
    questions = build_top200()
    out_path = Path(__file__).parent / "top200_questions.json"
    out_path.write_text(json.dumps(questions, indent=2), encoding="utf-8")

    print(f"Generated {len(questions)} questions")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
