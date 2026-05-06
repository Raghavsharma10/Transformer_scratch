def reduce_base(amount: int, base: int) -> tuple:
    """
    Compute the reduced base of the given parameters

    :param amount: the amount value
    :param base: current base value

    :return: tuple containing computed (amount, base)
    """
    if amount == 0:
        return 0, 0

    next_amount = amount
    next_base = base
    next_amount_is_integer = True
    while next_amount_is_integer:
        amount = next_amount
        base = next_base
        if next_amount % 10 == 0:
            next_amount = int(next_amount / 10)
            next_base += 1
        else:
            next_amount_is_integer = False

    return int(amount), int(base)