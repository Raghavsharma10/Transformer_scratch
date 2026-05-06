def add_firewall_rule(self, direction, action, src=None, dst=None):
        """Adds a firewall rule to the router.

        The TunTap router includes a very simple firewall for governing vassal's traffic.
        The first matching rule stops the chain, if no rule applies, the policy is "allow".

        :param str|unicode direction: Direction:

            * in
            * out

        :param str|unicode action: Action:

            * allow
            * deny

        :param str|unicode src: Source/mask.

        :param str|unicode dst: Destination/mask

        """
        value = [action]

        if src:
            value.extend((src, dst))

        self._set_aliased('router-firewall-%s' % direction.lower(), ' '.join(value), multi=True)

        return self