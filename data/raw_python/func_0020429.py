def get_builder_openshift_url(self):
        """ url of OpenShift where builder will connect """
        key = "builder_openshift_url"
        url = self._get_deprecated(key, self.conf_section, key)
        if url is None:
            logger.warning("%r not found, falling back to get_openshift_base_uri()", key)
            url = self.get_openshift_base_uri()
        return url