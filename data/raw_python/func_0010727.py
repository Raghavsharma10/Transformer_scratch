def exchange_currency(
        base: T, to_currency: str, *, conversion_rate: Decimal=None) -> T:
    """
    Exchanges Money, TaxedMoney and their ranges to the specified currency.
    get_rate parameter is a callable taking single argument (target currency)
    that returns proper conversion rate
    """
    if base.currency == to_currency:
        return base
    if base.currency != BASE_CURRENCY and to_currency != BASE_CURRENCY:
        # Exchange to base currency first
        base = exchange_currency(base, BASE_CURRENCY)

    if conversion_rate is None:
        conversion_rate = get_conversion_rate(base.currency, to_currency)

    if isinstance(base, Money):
        return Money(base.amount * conversion_rate, currency=to_currency)
    if isinstance(base, MoneyRange):
        return MoneyRange(
            exchange_currency(
                base.start, to_currency, conversion_rate=conversion_rate),
            exchange_currency(
                base.stop, to_currency, conversion_rate=conversion_rate))
    if isinstance(base, TaxedMoney):
        return TaxedMoney(
            exchange_currency(
                base.net, to_currency, conversion_rate=conversion_rate),
            exchange_currency(
                base.gross, to_currency, conversion_rate=conversion_rate))
    if isinstance(base, TaxedMoneyRange):
        return TaxedMoneyRange(
            exchange_currency(
                base.start, to_currency, conversion_rate=conversion_rate),
            exchange_currency(
                base.stop, to_currency, conversion_rate=conversion_rate))

    # base.currency was set but we don't know how to exchange given type
    raise TypeError('Unknown base for exchange_currency: %r' % (base,))