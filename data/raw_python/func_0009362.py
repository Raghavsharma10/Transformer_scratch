def _match_view(self, method, route_params):
        """ Detect a view matching the query

        :param method: HTTP method
        :param route_params: Route parameters dict
        :return: Method
        :rtype: Callable|None
        """
        method = method.upper()
        route_params = frozenset(k for k, v in route_params.items() if v is not None)

        for view_name, info in self.methods_map[method].items():
            if info.matches(method, route_params):
                return getattr(self, view_name)
        else:
            return None