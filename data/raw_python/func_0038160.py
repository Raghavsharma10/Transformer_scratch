def get_client_id(self):
        """ Attempt to get client_id from soundcloud homepage. """
        # FIXME: This method doesn't works
        id = re.search(
            "\"clientID\":\"([a-z0-9]*)\"",
            self.send_request(self.SC_HOME).read().decode("utf-8"))

        if not id:
            raise serror("Cannot retrieve client_id.")

        return id.group(1)