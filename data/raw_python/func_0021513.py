def get_rate_source():
    """Get the default Rate Source and return it."""
    backend = money_rates_settings.DEFAULT_BACKEND()
    try:
        return RateSource.objects.get(name=backend.get_source_name())
    except RateSource.DoesNotExist:
        raise CurrencyConversionException(
            "Rate for %s source do not exists. "
            "Please run python manage.py update_rates" % backend.get_source_name())