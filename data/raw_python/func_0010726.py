def get_conversion_rate(from_currency: str, to_currency: str) -> Decimal:
    """
    Get conversion rate to use in exchange
    """
    reverse_rate = False
    if to_currency == BASE_CURRENCY:
        # Fetch exchange rate for base currency and use 1 / rate for conversion
        rate_currency = from_currency
        reverse_rate = True
    else:
        rate_currency = to_currency
    rate = get_rate_from_db(rate_currency)

    if reverse_rate:
        conversion_rate = Decimal(1) / rate
    else:
        conversion_rate = rate
    return conversion_rate