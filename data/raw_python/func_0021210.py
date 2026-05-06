def with_preference_param(self):
        """Add the preference param to the ES request and return a new Search.

        The preference param avoids the bouncing effect with multiple
        replicas, documented on ES documentation.
        See: https://www.elastic.co/guide/en/elasticsearch/guide/current
        /_search_options.html#_preference for more information.
        """
        user_hash = self._get_user_hash()
        if user_hash:
            return self.params(preference=user_hash)
        return self