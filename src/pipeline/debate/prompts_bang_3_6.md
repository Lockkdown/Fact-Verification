# Prompts Template (for Table 3.6)

## Round 1 — Independent analysis (Parts-based)

```text
You are an EVIDENCE-BASED FACT-CHECKER. Analyze independently. Use ONLY the provided evidence. Do NOT use outside knowledge.

**Claim:** "{CLAIM}"
**Evidence:**
{EVIDENCE_BULLETS}

**STEP 1: BREAK CLAIM INTO KEY PARTS**
Identify 2-5 KEY verifiable parts of the claim.
(Key parts = core entities, time, location, quantity, or main event described by the claim)

For EACH part:
- Find a VERBATIM quote from evidence that covers it → status = COVERED
- If no quote found → status = MISSING, quote = NULL
- If evidence EXPLICITLY NEGATES the part (opposite value, mutually exclusive) → status = CONFLICT

**STEP 2: DECIDE**
- ALL key parts COVERED → **SUPPORTED**
- ANY key part CONFLICT → **REFUTED**
- Any key part MISSING → **NEI**

**RULES:**
- Quotes must be VERBATIM substrings copied from Evidence
- If you cannot find a verbatim quote, set quote = NULL
- CONFLICT only when evidence explicitly negates (not just missing info)

**Output JSON:**
{
    "parts": [
        {"part": "key part of claim", "status": "COVERED | MISSING | CONFLICT", "quote": "VERBATIM quote OR NULL"}
    ],
    "verdict": "SUPPORTED | REFUTED | NEI",
    "reasoning_vi": "1-2 câu giải thích"
}
```

### Notes (Implementation mapping)
- `Evidence` được đưa vào dưới dạng bullet list.
- `parts` là output structured field để trace claim→evidence.
- `verdict` dùng bộ nhãn: `SUPPORTED`, `REFUTED`, `NEI`.

---

## Round 2+ — Debate (Evidence-grounded rebuttals)

```text
**ROUND {ROUND_NUM}: DEBATE**

**Claim:** "{CLAIM}"
**Evidence:**
{EVIDENCE_BULLETS}

---

**Your previous position:** {OWN_PREV_VERDICT}
Your quote: "{OWN_PREV_QUOTE}"

**Other agents' positions:**
- {AGENT_NAME_1}: {AGENT_VERDICT_1} — "{AGENT_QUOTE_1}"
- {AGENT_NAME_2}: {AGENT_VERDICT_2} — "{AGENT_QUOTE_2}"

---

**TASK:**
Write a short research-style rebuttal.
1. State which parts of the claim you focus on.
2. Compare your evidence against other agents' evidence (agree/disagree and why).
3. Update your stance if and only if the evidence requires it.

**Evidence & Update Guidance:**
- Use **verbatim quotes** copied from the Evidence.
- Treat other agents' quotes as claims to be evaluated: is the quote relevant, and does it directly support/refute the claim?
- **Maintain your verdict by default.** Change your verdict only if you introduce a **new counter_quote** (verbatim) that directly undermines your previous quote/interpretation on a specific claim part.
- If you cannot find a clear, direct supporting/refuting quote for SUPPORTED/REFUTED, consider using NEI and briefly explain what is missing.

**Output JSON:**
{
    "key_parts_checked": ["key part 1...", "key part 2..."],
    "has_new_evidence": true,
    "has_strong_counterevidence": false,
    "rebuttals": [
        {
            "agent": "Agent name",
            "their_verdict": "SUPPORTED | REFUTED | NEI",
            "their_quote": "their quote",
            "issue": "irrelevant | not_direct | wrong_context | misses_key_part | weak_contradiction | no_supporting_quote",
            "counter_quote": "VERBATIM quote OR NULL"
        }
    ],
    "decision_change": "MAINTAIN | CONSIDER_CHANGE | CHANGE",
    "changed_from": "SUPPORTED | REFUTED | NEI | N/A",
    "change_trigger": "stronger_quote | missed_part | context_error | no_supporting_quote | N/A",
    "verdict": "SUPPORTED | REFUTED | NEI",
    "key_quote": "VERBATIM quote OR NULL",
    "reasoning_vi": "1-2 câu: giữ/đổi verdict vì parts + quote nào"
}
```

### Notes (Implementation mapping)
- `Your previous position` + `Other agents' positions` được build từ `previous_arguments`.
- Quote hiển thị được lấy từ `key_points` (đã được parser trích từ `parts`), ưu tiên tối đa 2 quotes để giữ prompt gọn.
- Round 2+ nhấn mạnh rule: **maintain by default**, chỉ change khi có **counter_quote** mới (verbatim).
