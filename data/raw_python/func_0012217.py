def set_bluetooth(self, active=None, name=None):
        """
        allows to activate/deactivate bluetooth and change the name
        """
        assert(active is not None or name is not None)

        log.debug("setting bluetooth state...")

        cmd, url = DEVICE_URLS["set_bluetooth"]
        json_data = {}
        if name is not None:
            json_data["name"] = name
        if active is not None:
            json_data["active"] = active

        return self._exec(cmd, url, json_data=json_data)