def int_to_words(number, european=False):
    """
    Converts an integer or float to words.

    .. code: python

        reusables.int_to_number(445)
        # 'four hundred forty-five'

        reusables.int_to_number(1.45)
        # 'one and forty-five hundredths'

    :param number: String, integer, or float to convert to words. The decimal
        can only be up to three places long, and max number allowed is 999
        decillion.
    :param european: If the string uses the european style formatting, i.e.
        decimal points instead of commas and commas instead of decimal points,
        set this parameter to True
    :return: The translated string
    """
    def ones(n):
        return "" if n == 0 else _numbers[n]

    def tens(n):
        teen = int("{0}{1}".format(n[0], n[1]))

        if n[0] == 0:
            return ones(n[1])
        if teen in _numbers:
            return _numbers[teen]
        else:
            ten = _numbers[int("{0}0".format(n[0]))]
            one = _numbers[n[1]]
            return "{0}-{1}".format(ten, one)

    def hundreds(n):
        if n[0] == 0:
            return tens(n[1:])
        else:
            t = tens(n[1:])
            return "{0} hundred {1}".format(_numbers[n[0]], "" if not t else t)

    def comma_separated(list_of_strings):
        if len(list_of_strings) > 1:
            return "{0} ".format("" if len(list_of_strings) == 2
                                 else ",").join(list_of_strings)
        else:
            return list_of_strings[0]

    def while_loop(list_of_numbers, final_list):
        index = 0
        group_set = int(len(list_of_numbers) / 3)
        while group_set != 0:
            value = hundreds(list_of_numbers[index:index + 3])
            if value:
                final_list.append("{0} {1}".format(value, _places[group_set])
                                  if _places[group_set] else value)
            group_set -= 1
            index += 3

    number_list = []
    decimal_list = []

    decimal = ''
    number = str(number)
    group_delimiter, point_delimiter = (",", ".") \
        if not european else (".", ",")

    if point_delimiter in number:
        decimal = number.split(point_delimiter)[1]
        number = number.split(point_delimiter)[0].replace(
            group_delimiter, "")
    elif group_delimiter in number:
        number = number.replace(group_delimiter, "")

    if not number.isdigit():
        raise ValueError("Number is not numeric")

    if decimal and not decimal.isdigit():
        raise ValueError("Decimal is not numeric")

    if int(number) == 0:
        number_list.append("zero")

    r = len(number) % 3
    d_r = len(decimal) % 3
    number = number.zfill(len(number) + 3 - r if r else 0)
    f_decimal = decimal.zfill(len(decimal) + 3 - d_r if d_r else 0)

    d = [int(x) for x in f_decimal]
    n = [int(x) for x in number]

    while_loop(n, number_list)

    if decimal and int(decimal) != 0:
        while_loop(d, decimal_list)

        if decimal_list:
            name = ''
            if len(decimal) % 3 == 1:
                name = 'ten'
            elif len(decimal) % 3 == 2:
                name = 'hundred'

            place = int((str(len(decimal) / 3).split(".")[0]))
            number_list.append("and {0} {1}{2}{3}ths".format(
                comma_separated(decimal_list), name,
                "-" if name and _places[place+1] else "", _places[place+1]))

    return comma_separated(number_list)