def api_url(self):
        """Returns the api_url or None.
        """
        if not self._api_url:
            error_msg = (
                f"Email is enabled but API_URL is not set. "
                f"See settings.{self.api_url_attr}"
            )
            try:
                self._api_url = getattr(settings, self.api_url_attr)
            except AttributeError:
                raise EmailNotEnabledError(error_msg, code="api_url_attribute_error")
            else:
                if not self._api_url:
                    raise EmailNotEnabledError(error_msg, code="api_url_is_none")
        return self._api_url