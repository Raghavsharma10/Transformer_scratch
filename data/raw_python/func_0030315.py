def update(self):
        """Cache the list into the data section of the record"""

        from ambry.orm.exc import NotFoundError
        from requests.exceptions import ConnectionError, HTTPError
        from boto.exception import S3ResponseError

        d = {}

        try:
            for k, v in self.list(full=True):
                if not v:
                    continue

                d[v['vid']] = {
                    'vid': v['vid'],
                    'vname': v.get('vname'),
                    'id': v.get('id'),
                    'name': v.get('name')
                }

            self.data['list'] = d
        except (NotFoundError, ConnectionError, S3ResponseError, HTTPError) as e:
            raise RemoteAccessError("Failed to update {}: {}".format(self.short_name, e))