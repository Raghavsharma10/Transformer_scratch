def set_handler(self, handler):
        """ Set transport handler

        @param handler: Handler, should derive from the
                        C{sockjs.cyclone.transports.base.BaseTransportMixin}
        """
        if self.handler is not None:
            raise Exception('Attempted to overwrite BaseSession handler')

        self.handler = handler
        self.transport_name = self.handler.name

        if self.conn_info is None:
            self.conn_info = handler.get_conn_info()
            self.stats.sessionOpened(self.transport_name)

        return True