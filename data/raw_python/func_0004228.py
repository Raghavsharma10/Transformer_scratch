def associate_interface_environments(self, int_env_map):
        """
        Method to add an interface.
        :param int_env_map: List containing interfaces and environments ids desired to be associates.
        :return: Id.
        """

        data = {'interface_environments': int_env_map}
        return super(ApiInterfaceRequest, self).post('api/v3/interface/environments/', data)