def get_expired(self):
        """
        Get all tokens which have expired.
        :return: All expired tokens.
        :rtype: :class:`esi.managers.TokenQueryset`
        """
        max_age = timezone.now() - timedelta(seconds=app_settings.ESI_TOKEN_VALID_DURATION)
        return self.filter(created__lte=max_age)