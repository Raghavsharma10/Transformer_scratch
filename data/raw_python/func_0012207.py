def send_notification(
        self, model, priority="warning", icon_type=None, lifetime=None
    ):
        """
        sends new notification to the device

        :param Model model: an instance of the Model class that should be used
        :param str priority: the priority of the notification
                             [info, warning or critical] (default: warning)
        :param str icon_type: the icon type of the notification
                              [none, info or alert] (default: None)
        :param int lifetime: the lifetime of the notification in ms
                             (default: 2 min)
        """
        assert(priority in ("info", "warning", "critical"))
        assert(icon_type in (None, "none", "info", "alert"))
        assert((lifetime is None) or (lifetime > 0))

        log.debug("sending notification...")

        cmd, url = DEVICE_URLS["send_notification"]

        json_data = {"model": model.json(), "priority": priority}

        if icon_type is not None:
            json_data["icon_type"] = icon_type
        if lifetime is not None:
            json_data["lifetime"] = lifetime

        return self._exec(cmd, url, json_data=json_data)