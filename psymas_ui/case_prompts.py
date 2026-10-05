"""Compact human-readable case-review defaults with sentence-local evidence."""

import hashlib
import json
from pathlib import Path

CASE_PROMPT_VERSION = 'case-review-evidence-v20-natural-summary'

LEGACY_DEFAULT_PROMPT_HASHES = {
    '449038515b3bc97f59e3a81bb228a54f66e703eeef2a544d93b4a963db64c40e',
    '64fed705e995c3673ff22aac9af17a66fee2957bdd72acd913c49fc22c543294',
    '36f8f01614464f9c0369459e66a733804a1d567fa051b12b0f417edec1187cab',
    'b688b47bf4c4a807d7053e58fdec2afb2fae4cbd6786addd067c0372f88f2ba7',
    '45a69478c5114ff4f3e918edd76c2abe8adf6f6f6991ee587cd6baa06af93188',
    '341c44c7905c7123ba89f5cd22df147509b93147b212607266a58a41718b8899',
    '0c2bc8f5879c21b7dad938ac9fcca53b43bdf533a14dba129225134e82496e41',
    '315427de682a74e8254678766ba0031e6f88d888a14546ccc4f9b07dc6e0b7e3',
    'f7b02caecd16d6e73129e724e16e2a831461399c3a291640e5733acee813411b',
}

_CONTRACT = """Write a human analyst's case summary using only this packet.
Use its exact priority, profile and domain strengths; do not recompute them. RT=timing, PK=exposed-item performance, TP=answer changes, MF=response-pattern fit. RT/PK/TP drive priority; MF supports it. SIM is context; CP is localization unless explicitly governed.
Only active governed domains support concerns. Raw cues cannot create a domain or raise strength. Never infer intent, prior access, cheating, guilt or sanctions.
Each evidence sentence needs its own allowed [E:DOMAIN/FAMILY] before punctuation; synthesis uses [E:DOMAIN/domain]. A named family needs its family tag. Each priority sentence needs the exact [P:label]; a mixed priority/strength sentence needs both tags. Copy allowed tags, never placeholders. Preserve the required JSON schema when supplied.
Name one eligible family per active domain. Flags are triggers, not magnitudes or probabilities; omit 1.0. Explain this only if essential.
Report both sides and direction of comparisons. Timing alone does not establish rapid responding. A PK flag alone does not establish advantage: compare exposed-item accuracy with the reference, including below-reference results. Preserve counts; never invent item lists or claim supplied data are missing.
Omit inactive behaviors and implementation details.
Lower non-exposed accuracy highlights a performance contrast; it does not itself weaken exposed-item advantage. Check item difficulty, reference comparability and legitimate exposure as possible explanations, not established facts.
"""

_STRUCTURE = """Use four sections with distinct roles:
1. Direct review conclusion: lead with the observed finding and key numerical contrast, at most 2 sentences; no priority or index names.
2. Evidence pattern: state exact priority ONCE with primary/supporting basis, at most 2 sentences; no repeated numbers.
3. Main index support: one cited sentence per point; group related families. Do not repeat numbers or priority. Multiple flags do not imply independent or additional evidence unless supported. Aim for 3 points; cover every active domain.
4. Recommended reviewer action: at most 2 checks, each naming what to inspect or compare and the question it resolves. Include a CP location only if its basis was explained earlier and changes a check.
Write short, plain-English sentences; each fact once. Apply rules silently, without repeating instructions or flag mechanics. Avoid vague 'review in context'. Evidence and citations take precedence over length.
"""

LOCAL_CASE_REVIEWER_PROMPT = (
    _CONTRACT + _STRUCTURE
    + "120-160 words excluding citations; start with section 1.\n\n{case_context}\n"
)

DEFAULT_CASE_REVIEWER_PROMPT = (
    _CONTRACT + _STRUCTURE
    + "Link primary families to the finding; separate observation from interpretation. Correction variants count as one family. "
      "150-200 words excluding citations; start with section 1.\n\n{case_context}\n"
)


def case_prompt_provenance(prompt: str, packet: str) -> dict:
    """Identify exact archived defaults from the actual sent prompt, including replays."""
    template = str(prompt).split('\n\nFINAL CITATION CHECK:', 1)[0]
    template = template.split('\n\nOUTPUT TRANSPORT:', 1)[0]
    if packet and packet in template:
        template = template.replace(packet, '{case_context}', 1)
    digest = hashlib.sha256(template.strip().encode()).hexdigest()
    result = dict(prompt_version='custom', prompt_variant='custom', prompt_template_hash=digest)
    for path in sorted((Path(__file__).parent / 'prompt_versions').glob('v*.json')):
        archive = json.loads(path.read_text(encoding='utf-8'))
        for variant, entry in archive['variants'].items():
            if digest == entry['sha256']:
                return dict(result, prompt_version=archive['version'], prompt_variant=variant)
    return result
