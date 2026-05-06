def _convert_method_settings_into_operations(method_settings=None):
    """Helper to handle the conversion of method_settings to operations

    :param method_settings:
    :return: list of operations
    """
    # operations docs here: https://tools.ietf.org/html/rfc6902#section-4
    operations = []
    if method_settings:
        for method in method_settings.keys():
            for key, value in method_settings[method].items():
                if isinstance(value, bool):
                    if value:
                        value = 'true'
                    else:
                        value = 'false'
                operations.append({
                    'op': 'replace',
                    'path': method + _resolve_key(key),
                    'value': value
                })
    return operations