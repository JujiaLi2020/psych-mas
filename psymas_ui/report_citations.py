"""Render case-bound evidence IDs; preserve raw drafts and explicit repairs."""
import json
import re
from .report_validation import case_citation_contract, TRAILING_SENTENCE_CITATIONS

SECTIONS = [('conclusion','Direct review conclusion'),('pattern','Evidence pattern'),
            ('support','Main index support'),('actions','Recommended reviewer action')]
SECTION_PURPOSES = {
    'conclusion': 'Plain-language concern and first check, without index names.',
    'pattern': 'Concise supplied priority basis, separating primary and supporting domains.',
    'support': 'Main eligible index families, their meaning, supplied comparisons and limitations; do not place all index support in pattern.',
    'actions': 'Concrete checks tied to the supplied evidence and available context.',
}


def is_structured_report(raw):
    text=str(raw or '').strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I).strip()
    try:
        return isinstance(json.loads(text),dict)
    except (ValueError,TypeError):
        return False


def citation_map(case_row, case_domains, case_trace):
    contract = case_citation_contract(case_row, case_domains, case_trace)
    tags = re.findall(r'\[(?:E:[A-Z]{2,3}/[^\]\n]+|P:[^\]\n]+)\]', contract)
    return {tag[1:-1]:tag for tag in tags}


def report_schema(case_row, case_domains, case_trace):
    ids = list(citation_map(case_row, case_domains, case_trace))
    sentence = {'type':'object','additionalProperties':False,
                'properties':{'text':{'type':'string','minLength':1},
                              'evidence_ids':{'type':'array','items':{'type':'string','enum':ids}}},
                'required':['text','evidence_ids']}
    return {'type':'object','additionalProperties':False,
            'properties':{key:{'type':'array','minItems':1,'items':sentence,
                               'description':SECTION_PURPOSES[key]} for key,_ in SECTIONS},
            'required':[key for key,_ in SECTIONS]}


def structured_report_instruction(case_row, case_domains, case_trace):
    ids = list(citation_map(case_row, case_domains, case_trace))
    return (
        '\n\nOUTPUT TRANSPORT: Return only a JSON object with arrays conclusion, pattern, support, actions. '
        'All four arrays MUST be nonempty, including support. Keep pattern to the priority synthesis; '
        'put index-family explanations, comparisons and limitations in support. '
        'Each array contains objects with nonblank text and evidence_ids. Each text is ONE sentence, without headings, '
        'bullet markers or bracketed citation tags. Select evidence_ids from the allowed list below; '
        'the application will render citations and section headings. A sentence stating both priority and '
        'domain strength requires both evidence and priority IDs; a strength-only sentence needs its domain ID. '
        'This JSON format replaces the prose formatting instructions. '
        'Separate different domains into separate sentences where possible. A mixed-domain sentence needs '
        'IDs for EVERY domain it discusses, not just the leading domain. In particular, exposed-item accuracy '
        'or timing comparisons belong to PK context even when discussed alongside an RT flag: cite E:PK/domain '
        'if allowed; cite an RT family separately for the RT screening claim. Do not cite inactive domains '
        'or turn a descriptive comparison into independent detector evidence. '
        'Check all sections and sentence-local IDs before returning JSON. '
        'Do not invent or spell new IDs. Allowed evidence_ids: '+json.dumps(ids)+'.\n'
    )


def prepare_report_text(raw, case_row, case_domains, case_trace):
    """Return audit prose, structural violations and exact repair provenance."""
    mapping = citation_map(case_row, case_domains, case_trace)
    text = str(raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I).strip()
    if text.startswith('{') or text.startswith('['):
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return '', ['invalid structured report JSON'], []
        if not isinstance(data, dict) or set(data) != {key for key,_ in SECTIONS}:
            return '', ['invalid structured report sections'], []
        violations=[]; parts=[]
        for number,(key,title) in enumerate(SECTIONS,1):
            parts.append(f'{number}. {title}')
            rows=data[key]
            if not isinstance(rows,list) or not rows:
                violations.append(f'empty or invalid structured section: {key}')
                continue
            for row in rows:
                if not isinstance(row,dict) or set(row) != {'text','evidence_ids'}:
                    violations.append('invalid structured sentence'); continue
                claim=row['text']; ids=row['evidence_ids']
                if not isinstance(claim,str) or not claim.strip() or not isinstance(ids,list):
                    violations.append('invalid structured sentence'); continue
                if re.search(r'\[(?:E:|P:)',claim,re.I):
                    violations.append('citation tags embedded in structured text')
                if any(not isinstance(item,str) or item not in mapping for item in ids):
                    violations.append('unknown structured evidence ID'); continue
                claim=claim.strip().rstrip('.!?')
                tags=' '.join(mapping[item] for item in dict.fromkeys(ids))
                prefix='- ' if key in {'support','actions'} else ''
                parts.append(prefix+claim+(' '+tags if tags else '')+'.')
            parts.append('')
        return '\n'.join(parts).strip(),sorted(set(violations)),[]

    # Explicitly known transcription error only. Do not fuzzy-match unknown
    # citations. Require a named family in that sentence and an active exact
    # target in the current case's allowlist.
    repairs=[]
    wrong='[E:PK/pk_l_lr_s]'; right='[E:PK/pk_lr_s]'
    if right in mapping.values():
        original=text
        text=TRAILING_SENTENCE_CITATIONS.sub(
            lambda match:' '+match.group(3).strip()+match.group(1)+' ',text)
        sentences=re.split(r'(?<=[.!?])\s+|\n+',text)
        for sentence in sentences:
            if wrong in sentence and re.search(r'(?<![\w])pk_LR_S(?![\w])',sentence,re.I):
                corrected=sentence.replace(wrong,right)
                text=text.replace(sentence,corrected,1)
                repairs.append({'from':wrong,'to':right,'reason':'explicit legacy typo; same-sentence family and active case match'})
        if not repairs:
            text=original
    return text,[],repairs
