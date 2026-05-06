def get_link(self, peer):
        """
        Retrieves the link to the given peer
        """
        for access in peer.accesses:
            if access.type == 'mqtt':
                break
        else:
            # No MQTT access found
            return None

        # Get server access tuple
        server = (access.server.host, access.server.port)

        with self.__lock:
            try:
                # Get existing link
                return self._links[server]

            except KeyError:
                # Create a new link
                link = self._links[server] = MQTTLink(access)
                return link