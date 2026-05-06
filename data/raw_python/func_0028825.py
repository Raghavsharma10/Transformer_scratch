def authenticate(self, bound_route, actual_params) -> bool:
        """
        Runs the pre-defined authenticaton service
        :param bound_route str route matched
        :param actual_params dict actual url parameters
        :rtype: bool
        """
        if self.__auth_service is not None:
            auth_route = "{0}_{1}{2}".format(self.__method, self.__route, bound_route)
            auth_data = self.__auth_service.authenticate(self.__request, auth_route, actual_params)
            if auth_data is True:
                self.app.auth_data = self.__auth_service.auth_data
            else:
                return False

        return True