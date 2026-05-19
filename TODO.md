# TODO

## Design Direction: Evidence, Not Verdicts

The engine should not treat every window as a forced Human/AI verdict. Some text is genuinely weak evidence because human and AI distributions overlap in the observable 4D engine space. The product and engine should represent that explicitly.

Core principle:

- Per-window scores are evidence.
- Document-level results are aggregates over evidence.
- High-confidence local windows can supply hard evidence.
- Low-confidence or overlapping windows should remain soft or ambiguous.
- The UI should make this visible without exposing internal engine machinery.

## 1. Window-Level Evidence Model

- Define a public window score object that separates:
  - `p_ai`
  - `p_human`
  - posterior margin
  - geometric confidence
  - accepted/rejected by high-confidence gate
  - evidence class: `hard_ai`, `hard_human`, `soft_ai`, `soft_human`, `ambiguous`
- Stop using a single per-window label as if it were a final verdict.
- Keep internal expert names, votes, and model geometry out of the public API by default.
- Add tests proving that ambiguous windows do not become hard verdicts just because `p_ai` is barely above or below 50%.

Acceptance criteria:

- A window can be scored without producing a hard Human/AI verdict.
- Hard evidence requires both posterior direction and sufficient geometric confidence.
- Public payloads expose scores and evidence class, not expert internals.

## 2. Document Aggregation Beyond Mean Score

Long documents should aggregate the distribution of window evidence, not just average probabilities.

Implement document-level metrics:

- mean `p_ai` and `p_human`
- median `p_ai`
- max `p_ai`
- min `p_ai`
- variance / spread of `p_ai`
- fraction of hard-AI windows
- fraction of hard-human windows
- fraction of ambiguous windows
- longest consecutive hard-AI run
- longest consecutive hard-human run
- strongest hard-AI window
- strongest hard-human window
- margin-weighted AI evidence
- margin-weighted human evidence
- geometric-confidence-weighted AI evidence
- geometric-confidence-weighted human evidence

Acceptance criteria:

- Document score includes both average probability and evidence-distribution summaries.
- A document with one strong AI window and many ambiguous windows is distinguishable from a uniformly weak 51% AI document.
- Tests cover mixed documents, repeated boilerplate, and contradictory windows.

## 3. Hard Evidence Gate

Use low-coverage/high-accuracy geometric gates to identify hard evidence windows, but do not treat the gate as the whole engine.

Tasks:

- Calibrate candidate hard-evidence thresholds on guardrails:
  - geometric confidence threshold
  - alignment threshold
  - posterior margin threshold
- Track separate thresholds for `hard_ai` and `hard_human` if false negatives and false positives behave asymmetrically.
- Report coverage, accepted accuracy, FNR, FPR, and accepted false positives/false negatives for each threshold pair.
- Add regression tests for known guardrails:
  - Declaration should not produce hard-AI evidence.
  - Moby-Dick/Jane Eyre should not produce hard-AI evidence from whole-document averaging artifacts.
  - Known generated samples should surface hard-AI evidence when the signal is present.

Acceptance criteria:

- Hard-evidence thresholds are configurable.
- The hard gate can abstain at window level without suppressing softer document evidence.
- Guardrail tests check both accepted hard evidence and rejected weak evidence.

## 4. Soft Evidence Layer

Ungated or weakly gated scores still carry information, but should be treated as soft evidence.

Tasks:

- Define soft evidence as posterior direction with insufficient hard-gate confidence.
- Aggregate soft evidence separately from hard evidence.
- Track whether soft evidence is consistent across many windows or isolated/noisy.
- Add a document-level field like `soft_evidence_balance`.
- Test cases where many weak-AI windows should differ from one weak-AI window.

Acceptance criteria:

- Document output can say, in effect: strong hard evidence, weak distributed evidence, mixed evidence, or insufficient evidence.
- Soft evidence never overrides a large amount of contradictory hard evidence without an explicit aggregation rule.

## 5. Intrinsic Ambiguity Handling

Some text windows should be treated as intrinsically ambiguous: formal email, boilerplate, encyclopedia prose, short generic responses, and template-like writing.

Tasks:

- Add an ambiguity class derived from low posterior margin, low geometric confidence, or strong expert disagreement.
- Keep the old abstention logic as a stub only; ambiguity should come from evidence geometry and score distribution, not hand-written prose markers.
- Add tests for:
  - short formal email
  - repeated generic insult/question text
  - Wikipedia-style factual prose
  - policy-template text
  - boilerplate business response
- Document-level aggregation should report the fraction of ambiguous windows.

Acceptance criteria:

- Ambiguous windows are not forced into confident Human/AI labels.
- The UI can show that a document has insufficient authorship evidence in many windows.
- Ambiguity is measured, not triggered by hard-coded style markers.

## 6. Windowing And Renormalization

Different window sizes are different projections of the same document. We should preserve that structure instead of choosing one window size blindly.

Tasks:

- Score at multiple window sizes, at minimum:
  - 80 words
  - 120 words
  - 240 words
  - whole passage/document when short enough
- Use overlapping windows by default.
- Compare how evidence changes across scales:
  - hard evidence stable across scales
  - hard evidence only at short windows
  - hard evidence only at long windows
  - evidence flips across scales
- Add a scale-consistency summary to document output.

Acceptance criteria:

- A document can be scored at multiple scales with comparable output.
- Tests verify that repeated text and long appended text do not create truncation artifacts.
- Aggregation can identify localized AI-like passages even when the document average looks human.

## 7. UI Requirements

The frontend should let users test the detector without exposing the MoE internals.

Tasks:

- Show document-level result as an evidence summary, not just a binary verdict.
- Show window ranking:
  - strongest AI evidence windows
  - strongest human evidence windows
  - ambiguous windows
- Show coverage:
  - hard evidence coverage
  - soft evidence coverage
  - ambiguous coverage
- Avoid internal wording like expert names, hidden engine, guardrails, or MoE internals.
- Provide enough paragraph/window detail for users to see why a long document score happened.

Acceptance criteria:

- Users can inspect which passages drive the document score.
- The UI does not claim certainty for intrinsically ambiguous text.
- The UI does not expose concealed MoE internals.

## 8. Testing Backlog

Add test suites around behavior, not just raw accuracy.

Required guardrail tests:

- Declaration of Independence:
  - whole document
  - named passages
  - overlapping windows
  - must not produce hard-AI evidence
- Moby-Dick:
  - whole document
  - passages
  - windows
  - false-positive hard-AI calls should fail tests
- Jane Eyre:
  - whole document
  - passages
  - windows
  - should not be more AI-like than known generated samples under hard evidence
- Wikipedia:
  - human intros
  - generated intros
  - windows
  - track false positives separately from soft ambiguity
- AI controls:
  - bundled GPT samples
  - HC3-style answers
  - Wiki generated samples
  - humanized generated text when available

Metrics to assert or report:

- full-document accuracy
- accepted hard-evidence accuracy
- hard-evidence coverage
- accepted FNR
- accepted FPR
- soft-evidence balance
- ambiguity fraction
- ROC/AUC from raw probabilities
- calibration curve from geometric confidence

## 9. Engineering Cleanup

- Keep `reports/` and `visualizations/` uncommitted unless they are explicitly used in docs or reproducibility evidence.
- Move reusable report logic into tested modules instead of one-off scripts.
- Keep experimental scripts out of production paths unless promoted deliberately.
- Add CLI commands for:
  - scoring a document with window evidence
  - running guardrail evaluation
  - sweeping hard-evidence thresholds
  - exporting a compact JSON result for the UI
- Make CI run the evidence aggregation tests without requiring external dataset downloads.
