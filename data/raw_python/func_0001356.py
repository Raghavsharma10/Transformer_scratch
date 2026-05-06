def shred(key_name: str,
          value: t.Any,
          field_names: t.Iterable[str] = SHRED_DATA_FIELD_NAMES) -> t.Union[t.Any, str]:
    """
    Replaces sensitive data in ``value`` with ``*`` if ``key_name`` contains something that looks like a secret.

    :param field_names: a list of key names that can possibly contain sensitive data
    :param key_name: a key name to check
    :param value: a value to mask
    :return: an unchanged value if nothing to hide, ``'*' * len(str(value))`` otherwise
    """
    key_name = key_name.lower()
    need_shred = False
    for data_field_name in field_names:
        if data_field_name in key_name:
            need_shred = True
            break

    if not need_shred:
        return value

    return '*' * len(str(value))