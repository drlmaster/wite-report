# Task Packet

## Scope

Create a Chinese Markdown reading report for `APT.pdf`.

## Files to Read

- `APT.pdf`
- `/tmp/APT.txt`
- `memoryVLA_reading_notes.md` for local report style reference

## Files Allowed to Edit

- `APT_reading_report.md`
- `plan/project-overview.md`
- `plan/outline.md`
- `plan/progress.md`
- `plan/task-packets/apt-reading-report.md`

## Required Skills

- `research-writing-assistant`
- `using-research-writing`
- `paper-orchestration`
- `verification`
- `bat-gr00t-skill`

## Evidence/Data Inputs

- PDF metadata from `pdfinfo`
- Full text extracted by `pdftotext -layout`
- Tables, figures, method sections, appendix details, and limitations extracted from the PDF text

## Required Artifacts

- A readable Markdown report with concrete experimental numbers and project-facing interpretation.

## Rejection Checks

- Do not invent references or results not present in the PDF.
- Do not claim the method solves long-horizon memory, because the paper lists this as a limitation.
- Do not present project adaptation ideas as already validated.

## Validation Commands

- `test -f APT_reading_report.md`
- `wc -l APT_reading_report.md`
- `rg -n "APT|VA prior|VLA likelihood|LIBERO-PRO|Pick-Place|局限|BTA|Capability-use audit" APT_reading_report.md plan/progress.md`

