def tense_id(*args, **kwargs):
    """ Returns the tense id for a given (tense, person, number, mood, aspect, negated).
        Aliases and compound forms (e.g., IMPERFECT) are disambiguated.
    """
    # Unpack tense given as a tuple, e.g., tense((PRESENT, 1, SG)):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
         if args[0] not in ((PRESENT, PARTICIPLE), (PAST, PARTICIPLE)):
             args = args[0]
    # No parameters defaults to tense=INFINITIVE, tense=PRESENT otherwise.
    if len(args) == 0 and len(kwargs) == 0:
        t = INFINITIVE
    else:
        t = PRESENT
    # Set default values.
    tense   = kwargs.get("tense"  , args[0] if len(args) > 0 else t)
    person  = kwargs.get("person" , args[1] if len(args) > 1 else 3) or None
    number  = kwargs.get("number" , args[2] if len(args) > 2 else SINGULAR)
    mood    = kwargs.get("mood"   , args[3] if len(args) > 3 else INDICATIVE)
    aspect  = kwargs.get("aspect" , args[4] if len(args) > 4 else IMPERFECTIVE)
    negated = kwargs.get("negated", args[5] if len(args) > 5 else False)
    # Disambiguate wrong order of parameters.
    if mood in (PERFECTIVE, IMPERFECTIVE):
        mood, aspect = INDICATIVE, mood
    # Disambiguate INFINITIVE.
    # Disambiguate PARTICIPLE, IMPERFECT, PRETERITE.
    # These are often considered to be tenses but are in fact tense + aspect.
    if tense == INFINITIVE:
        person = number = mood = aspect = None; negated=False
    if tense in ((PRESENT, PARTICIPLE), PRESENT+PARTICIPLE, PARTICIPLE, GERUND):
        tense, aspect = PRESENT, PROGRESSIVE
    if tense in ((PAST, PARTICIPLE), PAST+PARTICIPLE):
        tense, aspect = PAST, PROGRESSIVE
    if tense == IMPERFECT:
        tense, aspect = PAST, IMPERFECTIVE
    if tense == PRETERITE:
        tense, aspect = PAST, PERFECTIVE
    if aspect in (CONTINUOUS, PARTICIPLE, GERUND):
        aspect = PROGRESSIVE
    if aspect == PROGRESSIVE:
        person = number = None
    # Disambiguate CONDITIONAL.
    # In Spanish, the conditional is regarded as an indicative tense.
    if tense == CONDITIONAL and mood == INDICATIVE:
        tense, mood = PRESENT, CONDITIONAL
    # Disambiguate aliases: "pl" =>
    # (PRESENT, None, PLURAL, INDICATIVE, IMPERFECTIVE, False).
    return TENSES_ID.get(tense.lower(),
           TENSES_ID.get((tense, person, number, mood, aspect, negated)))