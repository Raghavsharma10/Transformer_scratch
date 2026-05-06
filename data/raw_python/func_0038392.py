def create_feature_type_rates(self, created=False):
        """
        If the role is being created we want to populate a rate for all existing feature_types.
        """
        if created:
            for feature_type in FeatureType.objects.all():
                FeatureTypeRate.objects.create(role=self, feature_type=feature_type, rate=0)