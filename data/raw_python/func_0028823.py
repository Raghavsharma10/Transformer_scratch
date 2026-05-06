def exec_route_callback(self) -> Response or object:
        """
        Executes the resolved end-point callback, or its fallback
        :rtype: Response or object
        """
        if self.__method.lower() in self.__end_points:
            for bound in self.__end_points[self.__method.lower()]:

                route = list(bound)[0]
                expected_params = self.get_url_params(route)
                actual_params = self.get_url_params(self.get_end_point_uri())

                if self.request_matches_route(self.get_end_point_uri(), route):
                    self.app.log.info("> {0} {1}".format(self.__method, self.__uri))
                    if self.authenticate(route, actual_params):
                        self.app.log.debug(
                            "%s(%d) %s" % ("body ", len(self.__request.body), self.__request.body.decode('utf-8'))
                        )
                        pattern_params = self.get_callback_pattern(expected_params, actual_params)
                        self.app.request_handle_time = (
                            lambda d: int((d.days * 24 * 60 * 60 * 1000) + (d.seconds * 1000) + (d.microseconds / 1000))
                        )(datetime.now() - self.__request_start)

                        return bound[route](self.__request, self.app, **pattern_params)
                    else:
                        raise AuthException("Authentication failed.")

        if self.__method == "OPTIONS":
            self.app.log.info("Route matched: {0} {1}".format(self.__method, self.__uri))
            return self.default_route_options()

        if self.__route == '' and self.__uri == '':
            return self.welcome_page()
        else:
            return self.no_route_found(self.__request)