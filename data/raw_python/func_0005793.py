def register_route(self, route_rules, label=None):
        """Registers a routing rule.

        :param RouteRule|list[RouteRule] route_rules:

        :param str|unicode label: Label to mark the given set of rules.
            This can be used in conjunction with ``do_goto`` rule action.

            * http://uwsgi.readthedocs.io/en/latest/InternalRouting.html#goto

        """
        route_rules = listify(route_rules)

        if route_rules and label:
            self._set(route_rules[0].command_label, label, multi=True)

        for route_rules in route_rules:
            self._set(route_rules.command, route_rules.value, multi=True)

        return self._section