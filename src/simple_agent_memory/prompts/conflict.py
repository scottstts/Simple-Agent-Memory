DETECT_CONFLICT = """Determine whether the new fact should REPLACE any existing facts about the same subject and predicate.
Answer YES if the new fact replaces any existing fact (i.e., they cannot both be true now).
Answer NO if they can coexist or if you are unsure.

Subject: {subject}
Predicate: {predicate}
Existing facts:
{existing_facts}

New fact: {subject} {predicate} {new_value}

Answer with exactly YES or NO. Nothing else."""
