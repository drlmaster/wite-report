# Progress

## 2026-06-23

- Stage: S5 Review / reading-report synthesis.
- Extracted `APT.pdf` with `pdftotext -layout` into `/tmp/APT.txt`.
- Read core sections: abstract, introduction, related work, method, experiments, conclusion, and appendices.
- Prepared task packet for `APT_reading_report.md`.
- Added clarification that APT Stage 1 is visually conditioned imitation learning similar to ACT/Diffusion Policy, but used as VA-prior pretraining for a VLA action expert rather than as the final policy.

### Capability-use audit

- Required skills: `research-writing-assistant`, `using-research-writing`, `paper-orchestration`, `verification`, `bat-gr00t-skill`.
- Skills actually used: all listed above.
- Inputs consumed: `APT.pdf`, `/tmp/APT.txt`, `memoryVLA_reading_notes.md`, PDF metadata, selected project skill guidance.
- Inputs not used and why: no external web sources used because the user provided the PDF and the report is grounded in that local source.
- Artifacts produced: `APT_reading_report.md`, `plan/project-overview.md`, `plan/outline.md`, `plan/task-packets/apt-reading-report.md`, `plan/progress.md`.
- Verification run: `test -f APT_reading_report.md && wc -l APT_reading_report.md plan/progress.md`; `rg -n "APT|VA prior|VLA likelihood|LIBERO-PRO|Pick-Place|局限|BTA|ACT|模仿学习|visual shortcut|Capability-use audit" APT_reading_report.md plan/progress.md`.
- Remaining risk: PDF text extraction can distort figure/table formatting, so figure-based values were summarized only when the text/table content was clear.
