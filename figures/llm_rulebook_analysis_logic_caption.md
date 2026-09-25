**Figure 1**  
*Rulebook-grounded LLM analysis in PsyMAS*

The PsyMAS language model does not determine which outputs count as evidence. Package outputs and selected raw-data summaries are first processed by the deterministic rulebook. The resolver aggregates correction variants into family-level signals, assigns evidence domains, applies B3 strength rules, and derives the case priority basis. Only the resulting case-specific evidence packet is provided to the LLM. The model explains governed evidence and drafts cautious reviewer guidance. An evidence-language audit checks the draft against the active domains, eligible families, and reporting constraints before human review. Raw-data summaries clarify governed evidence but cannot create flags or change evidence strength.

*Note.* The LLM cannot compute forensic indices, change thresholds, create evidence, infer intent, determine misconduct, or recommend sanctions.
