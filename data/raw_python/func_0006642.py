def T4(word, rules):
    '''Optionally split /u,y/-final diphthongs that do not take primary stress.
    E.g., [lau.ka.us], [va.ka.ut.taa].'''
    WORD = re.split(
        r'([ieaouäöy]+[^ieaouäöy]+\.*[ieaoäö]{1}(?:u|y)(?:\.*[^ieaouäöy]+|$))',  # noqa
        word, flags=re.I | re.U)

    PARTS = [[] for part in range(len(WORD))]

    for i, v in enumerate(WORD):

        if i != 0:
            vv = u_y_final_diphthongs(v)

            if vv:
                I = vv.start(1) + 1
                PARTS[i].append(v[:I] + '.' + v[I:])

        # include original form (non-application of rule)
        PARTS[i].append(v)

    WORDS = [w for w in product(*PARTS)]

    for WORD in WORDS:
        WORD = ''.join(WORD)
        RULES = rules + ' T4' if word != WORD else rules

        yield WORD, RULES