def get_rate_from_db(currency: str) -> Decimal:
    """
    Fetch currency conversion rate from the database
    """
    from .models import ConversionRate
    try:
        rate = ConversionRate.objects.get_rate(currency)
    except ConversionRate.DoesNotExist:  # noqa
        raise ValueError('No conversion rate for %s' % (currency, ))
    return rate.rate