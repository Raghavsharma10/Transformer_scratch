def value_compare(left, right, ordering=1):
    """
    SORT VALUES, NULL IS THE LEAST VALUE
    :param left: LHS
    :param right: RHS
    :param ordering: (-1, 0, 1) TO AFFECT SORT ORDER
    :return: The return value is negative if x < y, zero if x == y and strictly positive if x > y.
    """

    try:
        ltype = left.__class__
        rtype = right.__class__

        if ltype in list_types or rtype in list_types:
            if left == None:
                return ordering
            elif right == None:
                return - ordering

            left = listwrap(left)
            right = listwrap(right)
            for a, b in zip(left, right):
                c = value_compare(a, b) * ordering
                if c != 0:
                    return c

            if len(left) < len(right):
                return - ordering
            elif len(left) > len(right):
                return ordering
            else:
                return 0

        if ltype is float and isnan(left):
            left = None
            ltype = none_type
        if rtype is float and isnan(right):
            right = None
            rtype = none_type

        null_order = ordering*10
        ltype_num = TYPE_ORDER.get(ltype, null_order)
        rtype_num = TYPE_ORDER.get(rtype, null_order)

        type_diff = ltype_num - rtype_num
        if type_diff != 0:
            return ordering if type_diff > 0 else -ordering

        if ltype_num == null_order:
            return 0
        elif ltype is builtin_tuple:
            for a, b in zip(left, right):
                c = value_compare(a, b)
                if c != 0:
                    return c * ordering
            return 0
        elif ltype in data_types:
            for k in sorted(set(left.keys()) | set(right.keys())):
                c = value_compare(left.get(k), right.get(k)) * ordering
                if c != 0:
                    return c
            return 0
        elif left > right:
            return ordering
        elif left < right:
            return -ordering
        else:
            return 0
    except Exception as e:
        Log.error("Can not compare values {{left}} to {{right}}", left=left, right=right, cause=e)