def get_closing_rule_for_now(location):
    """
    Returns QuerySet of ClosingRules that are currently valid
    """
    now = get_now()

    if location:
        return ClosingRules.objects.filter(company=location,
                                           start__lte=now, end__gte=now)

    return Company.objects.first().closingrules_set.filter(start__lte=now,
                                                           end__gte=now)