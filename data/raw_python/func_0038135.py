def require_valid(self):
        """
        Ensures all tokens are still valid. If expired, attempts to refresh.
        Deletes those which fail to refresh or cannot be refreshed.
        :return: All tokens which are still valid.
        :rtype: :class:`esi.managers.TokenQueryset`
        """
        expired = self.get_expired()
        valid = self.exclude(pk__in=expired)
        valid_expired = expired.bulk_refresh()
        return valid_expired | valid