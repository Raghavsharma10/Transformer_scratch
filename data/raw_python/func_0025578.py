def save(self, *args, **kwargs):
        """Adds a subscription for the given user to the given object."""
        method_kwargs = self._get_method_kwargs()
        try:
            subscription = Subscription.objects.get(**method_kwargs)
        except Subscription.DoesNotExist:
            subscription = Subscription.objects.create(**method_kwargs)
        return subscription