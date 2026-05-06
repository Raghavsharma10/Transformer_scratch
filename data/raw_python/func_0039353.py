def convertbase(number, base=10):
    """
    Convert a number in base 10 to another base

    :type number: number
    :param number: The number to convert

    :type base: integer
    :param base: The base to convert to.
    """

    integer = number
    if not integer:
        return '0'
    sign = 1 if integer > 0 else -1
    alphanum = string.digits + string.ascii_lowercase
    nums = alphanum[:base]
    res = ''
    integer *= sign
    while integer:
        integer, mod = divmod(integer, base)
        res += nums[mod]
    return ('' if sign == 1 else '-') + res[::-1]