def stts2universal(token, tag):
    """ Converts an STTS tag to a universal tag.
        For example: ohne/APPR => ohne/PREP
    """
    if tag in ("KON", "KOUI", "KOUS", "KOKOM"):
        return (token, CONJ)
    if tag in ("PTKZU", "PTKNEG", "PTKVZ", "PTKANT"):
        return (token, PRT)
    if tag in ("PDF", "PDAT", "PIS", "PIAT", "PIDAT", "PPER", "PPOS", "PPOSAT"): 
        return (token, PRON)
    if tag in ("PRELS", "PRELAT", "PRF", "PWS", "PWAT", "PWAV", "PAV"):
        return (token, PRON)
    return penntreebank2universal(*stts2penntreebank(token, tag))