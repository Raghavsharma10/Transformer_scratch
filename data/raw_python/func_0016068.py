def observable_timestamp_compare(instance):
    """Ensure cyber observable timestamp properties with a comparison
    requirement are valid.
    """
    for key, obj in instance['objects'].items():
        compares = enums.TIMESTAMP_COMPARE_OBSERVABLE.get(obj.get('type', ''), [])
        print(compares)
        for first, op, second in compares:
            comp = getattr(operator, op)
            comp_str = get_comparison_string(op)

            if first in obj and second in obj and \
                    not comp(obj[first], obj[second]):
                msg = "In object '%s', '%s' (%s) must be %s '%s' (%s)"
                yield JSONError(msg % (key, first, obj[first], comp_str, second, obj[second]),
                                instance['id'])