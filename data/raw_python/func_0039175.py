def registry_comparison(registry0, registry1):
    """Compares two dictionaries of registry keys returning their difference."""
    comparison = {'created_keys': {},
                  'deleted_keys': [],
                  'created_values': {},
                  'deleted_values': {},
                  'modified_values': {}}

    for key, info in registry1.items():
        if key in registry0:
            if info[1] != registry0[key][1]:
                created, deleted, modified = compare_values(
                    registry0[key][1], info[1])

                if created:
                    comparison['created_values'][key] = (info[0], created)
                if deleted:
                    comparison['deleted_values'][key] = (info[0], deleted)
                if modified:
                    comparison['modified_values'][key] = (info[0], modified)
        else:
            comparison['created_keys'][key] = info

    for key in registry0.keys():
        if key not in registry1:
            comparison['deleted_keys'].append(key)

    return comparison