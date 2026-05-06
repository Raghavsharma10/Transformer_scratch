def merge_roles(dominant_name, deprecated_name):
    """
    Merges a deprecated role into a dominant role.
    """
    dominant_qs = ContributorRole.objects.filter(name=dominant_name)
    if not dominant_qs.exists() or dominant_qs.count() != 1:
        return
    dominant = dominant_qs.first()
    deprecated_qs = ContributorRole.objects.filter(name=deprecated_name)
    if not deprecated_qs.exists() or deprecated_qs.count() != 1:
        return
    deprecated = deprecated_qs.first()

    # Update Rates
    if not dominant.flat_rates.exists() and deprecated.flat_rates.exists():
        flat_rate = deprecated.flat_rates.first()
        flat_rate.role = dominant
        flat_rate.save()

    if not dominant.hourly_rates.exists() and deprecated.hourly_rates.exists():
        hourly_rate = deprecated.hourly_rates.first()
        hourly_rate.role = dominant
        hourly_rate.save()

    for ft_rate in deprecated.feature_type_rates.all():
        dom_ft_rate = dominant.feature_type_rates.filter(feature_type=ft_rate.feature_type)

        if dom_ft_rate.exists() and dom_ft_rate.first().rate == 0:
            dom_ft_rate.first().delete()

        if not dom_ft_rate.exists():
            ft_rate.role = dominant
            ft_rate.save()

    # Update contributions
    for contribution in deprecated.contribution_set.all():
        contribution.role = dominant
        contribution.save()

    # Update overrides
    for override in deprecated.overrides.all():
        dom_override_qs = dominant.overrides.filter(contributor=override.contributor)
        if not dom_override_qs.exists():
            override.role = dominant
            override.save()
        else:
            dom_override = dom_override_qs.first()
            for flat_override in override.override_flatrate.all():
                flat_override.profile = dom_override
                flat_override.save()
            for hourly_override in override.override_hourly.all():
                hourly_override.profile = dom_override
                hourly_override.save()
            for feature_type_override in override.override_feature_type.all():
                feature_type_override.profile = dom_override
                feature_type_override.save()