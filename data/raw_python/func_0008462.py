def _suffix_rules(token, tag="NN"):
    """ Default morphological tagging rules for English, based on word suffixes.
    """
    if isinstance(token, (list, tuple)):
        token, tag = token
    if token.endswith("ing"):
        tag = "VBG"
    if token.endswith("ly"):
        tag = "RB"
    if token.endswith("s") and not token.endswith(("is", "ous", "ss")):
        tag = "NNS"
    if token.endswith(("able", "al", "ful", "ible", "ient", "ish", "ive", "less", "tic", "ous")) or "-" in token:
        tag = "JJ"
    if token.endswith("ed"):
        tag = "VBN"
    if token.endswith(("ate", "ify", "ise", "ize")):
        tag = "VBP"
    return [token, tag]