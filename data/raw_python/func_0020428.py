def get_openshift_base_uri(self):
        """
        https://<host>[:<port>]/

        :return: str
        """
        deprecated_key = "openshift_uri"
        key = "openshift_url"
        val = self._get_value(deprecated_key, self.conf_section, deprecated_key)
        if val is not None:
            warnings.warn("%r is deprecated, use %r instead" % (deprecated_key, key))
            return val
        return self._get_value(key, self.conf_section, key)