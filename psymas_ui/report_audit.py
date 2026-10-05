"""Evidence-bound LLM audit records, separate from display text."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
import pandas as pd
from .case_prompts import case_prompt_provenance

VALIDATOR_VERSION = 'case-report-audit-v7-pk-timing-limitations'


def evidence_fingerprint(case_row, case_domains, case_trace, case_auxiliary, context_facts=''):
    row_keys = ['Examinee_ID','Review_Priority','Evidence_Status','Review_Priority_Rule','B6_Code']
    row = {key:str(case_row.get(key, '')) for key in row_keys}
    def records(frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return []
        values=json.loads(frame.to_json(orient='records',date_format='iso'))
        return sorted(values,key=lambda value:json.dumps(value,sort_keys=True))
    payload={'case':row,'domains':records(case_domains),'trace':records(case_trace),
             'auxiliary':records(case_auxiliary),'context_facts':str(context_facts)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def make_audit_record(*, run_id, case_id, provider, model, prompt, packet, evidence_hash,
                      raw_text, display_text, violations, kind='interpretation'):
    return {'audit_id':str(uuid.uuid4()),'created_at':datetime.now(timezone.utc).isoformat(),
            'run_id':str(run_id),'case_id':str(case_id),'provider':str(provider),'model':str(model),
            'kind':kind,'validator_version':VALIDATOR_VERSION,'evidence_hash':evidence_hash,
            'prompt_hash':hashlib.sha256(prompt.encode()).hexdigest(),
            'packet_hash':hashlib.sha256(packet.encode()).hexdigest(),
            'prompt_text':prompt,'packet_text':packet,'raw_text':raw_text,'display_text':display_text,
            'violations':list(violations),'passed':not violations,
            **case_prompt_provenance(prompt, packet)}
