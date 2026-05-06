def get_proxy_config(self, headers, path):
        """
        stub. this really needs to be a call to the remote
        restful interface to get the appropriate host and
        headers to use for this upload
        """
        self.ofs.conn.add_aws_auth_header(headers, 'PUT', path)
        from pprint import pprint
        pprint(headers)
        host = self.ofs.conn.server_name()
        return host, headers