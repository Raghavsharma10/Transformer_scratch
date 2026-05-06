def timestamp_compare(instance):
    """Ensure timestamp properties with a comparison requirement are valid.

    E.g. `modified` must be later or equal to `created`.
    """
    compares = [('modified', 'ge', 'created')]
    additional_compares = enums.TIMESTAMP_COMPARE.get(instance.get('type', ''), [])
    compares.extend(additional_compares)

    for first, op, second in compares:
        comp = getattr(operator, op)
        comp_str = get_comparison_string(op)

        if first in instance and second in instance and \
                not comp(instance[first], instance[second]):
            msg = "'%s' (%s) must be %s '%s' (%s)"
            yield JSONError(msg % (first, instance[first], comp_str, second, instance[second]),
                            instance['id'])