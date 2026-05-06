def base_convert_money(amount, currency_from, currency_to):
    """
    Convert 'amount' from 'currency_from' to 'currency_to'
    """
    source = get_rate_source()

    # Get rate for currency_from.
    if source.base_currency != currency_from:
        rate_from = get_rate(currency_from)
    else:
        # If currency from is the same as base currency its rate is 1.
        rate_from = Decimal(1)

    # Get rate for currency_to.
    rate_to = get_rate(currency_to)

    if isinstance(amount, float):
        amount = Decimal(amount).quantize(Decimal('.000001'))

    # After finishing the operation, quantize down final amount to two points.
    return ((amount / rate_from) * rate_to).quantize(Decimal("1.00"))