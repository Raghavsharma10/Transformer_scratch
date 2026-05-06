def _set_bucket(self, type, name):
        """
        prepares bucket, sets index name
        :param str type: bucket type
        :param str name: bucket name
        :return:
        """
        if type:
            self._cfg['bucket_type'] = type
        if name:
            self._cfg['bucket_name'] = name
        self.bucket = self._client.bucket_type(self._cfg['bucket_type']
                                               ).bucket(self._cfg['bucket_name'])
        self.index_name = "%s_%s" % (self._cfg['bucket_type'], self._cfg['bucket_name'])
        return self