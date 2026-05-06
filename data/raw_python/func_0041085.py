def api_key(self):
        """Returns the api_key or None.
        """
        if not self._api_key:
            error_msg = (
                f"Email is enabled but API_KEY is not set. "
                f"See settings.{self.api_key_attr}"
            )
            try:
                self._api_key = getattr(settings, self.api_key_attr)
            except AttributeError:
                raise EmailNotEnabledError(error_msg, code="api_key_attribute_error")
            else:
                if not self._api_key:
                    raise EmailNotEnabledError(error_msg, code="api_key_is_none")
        return self._api_key