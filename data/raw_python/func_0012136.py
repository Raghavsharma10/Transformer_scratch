def _clean_tag(t):
    """Fix up some garbage errors."""
    # TODO: when score present, include info.
    t = _scored_patt.sub(string=t, repl='')
    if t == '_country_' or t.startswith('_country:'):
        t = 'nnp_country'
    elif t == 'vpb':
        t = 'vb'  # "carjack" is listed with vpb tag.
    elif t == 'nnd':
        t = 'nns'  # "abbes" is listed with nnd tag.
    elif t == 'nns_root:':
        t = 'nns'  # 'micros' is listed as nns_root.
    elif t == 'root:zygote':
        t = 'nn'  # 'root:zygote' for zygote. :-/
    elif t.startswith('root:'):
        t = 'uh'  # Don't know why, but these are all UH tokens.
    elif t in ('abbr_united_states_marine_corps', 'abbr_orange_juice'):
        t = "abbreviation"
    elif t == '+abbreviation':
        t = 'abbreviation'
    elif t.startswith('fw_misspelling:'):
        t = 'fw'
    return t