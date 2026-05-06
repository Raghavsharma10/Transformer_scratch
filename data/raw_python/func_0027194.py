def _create_historical_estimates(resource, configuration):
    """ Create consumption details and price estimates for past months.

        Usually we need to update historical values on resource import.
    """
    today = timezone.now()
    month_start = core_utils.month_start(today)
    while month_start > resource.created:
        month_start -= relativedelta(months=1)
        models.PriceEstimate.create_historical(resource, configuration, max(month_start, resource.created))