def expired(self):
        """Return boolean indicating token expiration."""
        now = timezone.now()
        if self.created < now - token_settings.EXPIRING_TOKEN_LIFESPAN:
            return True
        return False