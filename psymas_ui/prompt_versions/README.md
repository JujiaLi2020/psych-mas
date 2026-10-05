# Case review prompt versions

The active version is `CASE_PROMPT_VERSION` in `../case_prompts.py`.
Each JSON snapshot stores both complete default templates, character counts, and
SHA-256 hashes of the UTF-8 template after stripping surrounding whitespace.
Published snapshots must not be overwritten.

| Version | Change |
| --- | --- |
| v18: human-summary | Plain-language four-section report; 150–200 words hosted, 120–160 words local. |
| v19: focused-summary | Priority and numerical comparisons each appear once; lower non-exposed accuracy does not by itself weaken a contrast; CP actions require an explained basis. |
| v20: natural-summary | Lead with observations; apply instructions silently; group related flags without claiming independence; give concrete checks and questions. |

Generation audits retain the full sent prompt and its hash, plus `prompt_version`,
`prompt_variant`, and `prompt_template_hash`. Identification uses the actual sent
prompt and packet, not the currently selected default. Historical or edited
templates without an exact archived match are marked `custom`; existing SQLite
rows remain unchanged. Their full prompts remain available for audit.

For a new release, increment the active version, add a new JSON snapshot, and
include the previous defaults' hashes in `LEGACY_DEFAULT_PROMPT_HASHES`.
Verify snapshot integrity, migration, custom-edit preservation, and citations.
Template character limits are regression controls, not tokenizer counts or a
guarantee that an arbitrary evidence packet fits every model's context window.

To roll back, restore both template strings and the active version from the
desired snapshot; keep all snapshots and audit records. Include the replaced
defaults' hashes in migration so unchanged defaults can migrate, while user
edits remain untouched. Rollback affects future generation, not saved reports.
