def _get_from_riak(self, key):
        """
        Args:
            key (str): riak key
        Returns:
            (tuple): riak obj json data and riak key
        """

        obj = self.bucket.get(key)

        if obj.exists:
            return obj.data, obj.key

        raise ObjectDoesNotExist("%s %s" % (key, self.compiled_query))