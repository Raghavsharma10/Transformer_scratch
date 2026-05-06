def register_route(self, src, dst, gateway):
        """Adds a routing rule to the tuntap router.

        :param str|unicode src: Source/mask.

        :param str|unicode dst: Destination/mask.

        :param str|unicode gateway: Gateway address.

        """
        self._set_aliased('router-route', ' '.join((src, dst, gateway)), multi=True)

        return self