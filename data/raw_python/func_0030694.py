def system_image_type(self, system_image_type):
        """
        Sets the system_image_type of this BuildEnvironmentRest.

        :param system_image_type: The system_image_type of this BuildEnvironmentRest.
        :type: str
        """
        allowed_values = ["DOCKER_IMAGE", "VIRTUAL_MACHINE_RAW", "VIRTUAL_MACHINE_QCOW2", "LOCAL_WORKSPACE"]
        if system_image_type not in allowed_values:
            raise ValueError(
                "Invalid value for `system_image_type` ({0}), must be one of {1}"
                .format(system_image_type, allowed_values)
            )

        self._system_image_type = system_image_type