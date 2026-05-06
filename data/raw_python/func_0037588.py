def get_sponsored_special_coverages():
    """:returns: Django query for all active special coverages with active campaigns."""
    now = timezone.now()
    qs = SpecialCoverage.objects.filter(
        tunic_campaign_id__isnull=False,
    )
    return qs.filter(
        start_date__lte=now,
        end_date__gt=now
    ) | qs.filter(
        start_date__lte=now,
        end_date__isnull=True
    )