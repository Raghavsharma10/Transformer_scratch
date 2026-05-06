def device_add_rule(self, direction, action, src, dst, target=None):
        """Adds a tuntap device rule.

        To be used in a vassal.

        :param str|unicode direction: Direction:

            * in
            * out.

        :param str|unicode action: Action:

            * allow
            * deny
            * route
            * gateway.

        :param str|unicode src: Source/mask.

        :param str|unicode dst: Destination/mask.

        :param str|unicode target: Depends on action.

            * Route / Gateway: Accept addr:port

        """
        value = [direction, src, dst, action]

        if target:
            value.append(target)

        self._set_aliased('device-rule', ' '.join(value), multi=True)

        return self