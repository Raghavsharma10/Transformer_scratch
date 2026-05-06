def update_feature_type_rates(sender, instance, created, *args, **kwargs):
    """
    Creates a default FeatureTypeRate for each role after the creation of a FeatureTypeRate.
    """
    if created:
        for role in ContributorRole.objects.all():
            FeatureTypeRate.objects.create(role=role, feature_type=instance, rate=0)