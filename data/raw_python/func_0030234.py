def get_url_param(self, index, default=None):
        """
        Return url parameter with given index.

        Args:
        - index: starts from zero, and come after controller and
          action names in url.
        """
        params = self.get_url_params()
        return params[index] if index < len(params) else default