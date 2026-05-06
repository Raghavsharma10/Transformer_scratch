def unique_index(data, keys=None, fail_on_dup=True):
    """
    RETURN dict THAT USES KEYS TO INDEX DATA
    ONLY ONE VALUE ALLOWED PER UNIQUE KEY
    """
    o = UniqueIndex(listwrap(keys), fail_on_dup=fail_on_dup)

    for d in data:
        try:
            o.add(d)
        except Exception as e:
            o.add(d)
            Log.error(
                "index {{index}} is not unique {{key}} maps to both {{value1}} and {{value2}}",
                index=keys,
                key=select([d], keys)[0],
                value1=o[d],
                value2=d,
                cause=e,
            )
    return o