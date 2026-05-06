def _exec(self, cmd, url, json_data=None):
        """
        execute a command at the device using the RESTful API

        :param str cmd: one of the REST commands, e.g. GET or POST
        :param str url: URL of the REST API the command should be applied to
        :param dict json_data: json data that should be attached to the command
        """
        assert(cmd in ("GET", "POST", "PUT", "DELETE"))
        assert(self.dev is not None)

        if json_data is None:
            json_data = {}

        # add device address to the URL
        url = url.format(self.dev["ipv4_internal"])

        # set basic authentication
        auth = HTTPBasicAuth("dev", self.dev["api_key"])

        # execute HTTP request
        res = None
        if cmd == "GET":
            res = self._local_session.session.get(
                url, auth=auth, verify=False
            )

        elif cmd == "POST":
            res = self._local_session.session.post(
                url, auth=auth, json=json_data, verify=False
            )

        elif cmd == "PUT":
            res = self._local_session.session.put(
                url, auth=auth, json=json_data, verify=False
            )

        elif cmd == "DELETE":
            res = self._local_session.session.delete(
                url, auth=auth, verify=False
            )

        if res is not None:
            # raise an exception on error
            res.raise_for_status()

        return res.json()