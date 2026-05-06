def gender(word, pos=NOUN):
    """ Returns the gender (MALE, FEMALE or NEUTRAL) for nouns (majority vote).
        Returns None for words that are not nouns.
    """
    w = word.lower()
    if pos == NOUN:
        # Default rules (baseline = 32%).
        if w.endswith(gender_masculine):
            return MASCULINE
        if w.endswith(gender_feminine):
            return FEMININE
        if w.endswith(gender_neuter):
            return NEUTER
        # Majority vote.
        for g in gender_majority_vote:
            if w.endswith(gender_majority_vote[g]):
                return g