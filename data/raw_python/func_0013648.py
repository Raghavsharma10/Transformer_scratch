def compare_numeric(src_num, dst_num):
    """Compare numerical values. You can use '<%d','>%d'."""
    dst_num = float(dst_num)

    match = numeric_compare_regex.match(src_num)
    if not match:
        error = "Failed numeric comparison. Collected: {}. Expected: {}".format(dst_num, src_num)
        raise ValueError(error)

    operand = {
        "<": "__lt__",
        ">": "__gt__",
        ">=": "__ge__",
        "<=": "__le__",
        "==": "__eq__",
        "!=": "__ne__",
    }
    return getattr(dst_num, operand[match.group(1)])(float(match.group(2)))