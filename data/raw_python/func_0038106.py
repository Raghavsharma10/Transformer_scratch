def on_connect(self, client, userdata, flags, rc):
        """
        on_connect(self, client, obj, flags, rc):
        client
            the client instance for this callback
        userdata
            the private user data as set in Client() or userdata_set()
        flags
            response flags sent by the broker
        rc
            the connection result
        """
        _logger.debug("Connection established with result code %s" % rc)

        if self.reg_thread is not None and self.reg_thread.is_alive():
            _logger.debug("Joining previous reg_thread")
            self.reg_thread.join()

        def reg():
            delay = None
            if hasattr(self.reg_delay, '__call__'):
                delay = self.reg_delay()
            else:
                delay = self.reg_delay

            sleep(delay)
            self._conn.set_tunnels(self._conn.tunnels)
            model_profile = self.get_profile("model")
            view_profile = self.get_profile("view")
            self.deregister(model_profile)
            self.deregister(view_profile)
            self.register(model_profile)
            self.register(view_profile)
            self.is_ready.set()

        self.reg_thread = Thread(target=reg)
        self.reg_thread.daemon = True
        self.reg_thread.start()