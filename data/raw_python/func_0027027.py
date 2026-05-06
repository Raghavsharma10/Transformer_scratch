def has_user(self, user, role=None, timestamp=False):
        """
        Checks whether user has role in entity.
        `timestamp` can have following values:
            - False - check whether user has role in entity at the moment.
            - None - check whether user has permanent role in entity.
            - Datetime object - check whether user will have role in entity at specific timestamp.
        """
        permissions = self.permissions.filter(user=user, is_active=True)

        if role is not None:
            permissions = permissions.filter(role=role)

        if timestamp is None:
            permissions = permissions.filter(expiration_time=None)
        elif timestamp:
            permissions = permissions.filter(Q(expiration_time=None) | Q(expiration_time__gte=timestamp))

        return permissions.exists()