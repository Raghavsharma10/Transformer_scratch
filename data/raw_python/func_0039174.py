def compare_registries(fs0, fs1, concurrent=False):
    """Compares the Windows Registry contained within the two File Systems.

    If the concurrent flag is True,
    two processes will be used speeding up the comparison on multiple CPUs.

    Returns a dictionary.

        {'created_keys': {'\\Reg\\Key': (('Key', 'Type', 'Value'), ...)}
         'deleted_keys': ['\\Reg\\Key', ...],
         'created_values': {'\\Reg\\Key': (('Key', 'Type', 'NewValue'), ...)},
         'deleted_values': {'\\Reg\\Key': (('Key', 'Type', 'OldValue'), ...)},
         'modified_values': {'\\Reg\\Key': (('Key', 'Type', 'NewValue'), ...)}}

    """
    hives = compare_hives(fs0, fs1)

    if concurrent:
        future0 = concurrent_parse_registries(fs0, hives)
        future1 = concurrent_parse_registries(fs1, hives)

        registry0 = future0.result()
        registry1 = future1.result()
    else:
        registry0 = parse_registries(fs0, hives)
        registry1 = parse_registries(fs1, hives)

    return registry_comparison(registry0, registry1)