def _clean_dict(target_dict, whitelist=None):
    """ Convenience function that removes a dicts keys that have falsy values
    """
    assert isinstance(target_dict, dict)
    return {
        ustr(k).strip(): ustr(v).strip()
        for k, v in target_dict.items()
        if v not in (None, Ellipsis, [], (), "")
        and (not whitelist or k in whitelist)
    }