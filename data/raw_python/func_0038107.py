def register(self, reg_data, retry=True, interval=1, timeout=3):
        """
        register function
        retry
            True, infinity retries
            False, no retries
            Number, retries times
        interval
            time period for retry
        return
            False if no success
            Tunnel if success
        """
        if len(reg_data["resources"]) == 0:
            _logger.debug("%s no need to register due to no resources" %
                          (reg_data["name"]))
            return

        def _register():
            try:
                resp = self.publish.direct.post(
                    "/controller/registration", reg_data)
                if resp.code == 200:
                    return resp
            except TimeoutError:
                _logger.debug("Register message is timeout")

            return False

        resp = _register()
        while resp is False:
            _logger.debug("Register failed.")
            self.deregister(reg_data)
            resp = _register()

        if resp is None:
            _logger.error("Can\'t not register to controller")
            self.stop()
            return False

        self._conn.set_tunnel(
            reg_data["role"], resp.data["tunnel"], self.on_sanji_message)
        self.bundle.profile["currentTunnels"] = [
            tunnel for tunnel, callback in self._conn.tunnels.items()]
        self.bundle.profile["regCount"] = \
            self.bundle.profile.get("reg_count", 0) + 1

        _logger.debug("Register successfully %s tunnel: %s"
                      % (reg_data["name"], resp.data["tunnel"],))