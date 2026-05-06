def serve(self, config, path, route_name=None, permission=None, **view_options):
        """
        Serves this API from the inputted root path
        """
        route_name = route_name or path.replace('/', '.').strip('.')
        path = path.strip('/') + '*traverse'

        self.route_name = route_name
        self.base_permission = permission

        # configure the route and the path
        config.add_route(route_name, path, factory=self.factory)
        config.add_view(
            self.handle_standard_error,
            route_name=route_name,
            renderer='json2',
            context=StandardError
        ),
        config.add_view(
            self.handle_http_error,
            route_name=route_name,
            renderer='json2',
            context=HTTPException
        )
        config.add_view(
            self.process,
            route_name=route_name,
            renderer='json2',
            **view_options
        )