def active(self, registered_only=True):
        "Returns all active users, e.g. not logged and non-expired session."
        visitors = self.filter(
            expiry_time__gt=timezone.now(),
            end_time=None
        )
        if registered_only:
            visitors = visitors.filter(user__isnull=False)
        return visitors