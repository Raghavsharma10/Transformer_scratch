def buildconfig_update(orig, new, remove_nonexistent_keys=False):
    """Performs update of given `orig` BuildConfig with values from `new` BuildConfig.
    Both BuildConfigs have to be represented as `dict`s.

    This function:
    - adds all key/value pairs to `orig` from `new` that are missing
    - replaces values in `orig` for keys that are in both
    - removes key/value pairs from `orig` for keys that are not in `new`,
      but only in dicts nested inside `strategy` key
      (see https://github.com/projectatomic/osbs-client/pull/273#issuecomment-148038314)
    """
    if isinstance(orig, dict) and isinstance(new, dict):
        clean_triggers(orig, new)
        if remove_nonexistent_keys:
            missing = set(orig.keys()) - set(new.keys())
            for k in missing:
                orig.pop(k)
        for k, v in new.items():
            if k == 'strategy':
                remove_nonexistent_keys = True
            if isinstance(orig.get(k), dict) and isinstance(v, dict):
                buildconfig_update(orig[k], v, remove_nonexistent_keys)
            else:
                orig[k] = v