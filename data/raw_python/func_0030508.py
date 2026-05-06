def get_conn_info(self):
        from sockjs.cyclone.conn import ConnectionInfo

        """ Return C{ConnectionInfo} object from current transport """
        return ConnectionInfo(self.request.remote_ip,
                              self.request.cookies,
                              self.request.arguments,
                              self.request.headers,
                              self.request.path)