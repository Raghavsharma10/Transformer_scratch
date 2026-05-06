def authenticate_request(self, method, bucket='', key='', headers=None):
        '''Authenticate a HTTP request by filling in Authorization field header.

        :param method: HTTP method (e.g. GET, PUT, POST)
        :param bucket: name of the bucket.
        :param key: name of key within bucket.
        :param headers: dictionary of additional HTTP headers.

        :return: boto.connection.HTTPRequest object with Authorization header
        filled (NB: will also have a Date field if none before and a User-Agent
        field will be set to Boto).
        '''
        # following is extracted from S3Connection.make_request and the method
        # it calls: AWSAuthConnection.make_request
        path = self.conn.calling_format.build_path_base(bucket, key)
        auth_path = self.conn.calling_format.build_auth_path(bucket, key)
        http_request = boto.connection.AWSAuthConnection.build_base_http_request(
                self.conn,
                method,
                path,
                auth_path,
                {},
                headers
                )
        http_request.authorize(connection=self.conn)
        return http_request