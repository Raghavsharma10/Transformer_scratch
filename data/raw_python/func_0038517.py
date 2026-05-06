def dispatch(self, message):
        """
        dispatch
        """
        results = []
        # match routes
        for resource, route in self.routes.items():
            __message = message.match(route)
            if __message is None:
                continue

            route_result = route.dispatch(__message)
            if len(route_result) == 0:
                continue

            results.append({
                "handlers": route_result,
                "message": __message
            })

        return results