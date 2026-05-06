def update_rates(self):
        """
        Creates or updates rates for a source
        """
        source, created = RateSource.objects.get_or_create(name=self.get_source_name())
        source.base_currency = self.get_base_currency()
        source.save()

        for currency, value in six.iteritems(self.get_rates()):
            try:
                rate = Rate.objects.get(source=source, currency=currency)
            except Rate.DoesNotExist:
                rate = Rate(source=source, currency=currency)

            rate.value = value
            rate.save()