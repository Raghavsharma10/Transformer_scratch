def build_type(self, build_type):
        """
        Sets the build_type of this BuildConfigurationRest.

        :param build_type: The build_type of this BuildConfigurationRest.
        :type: str
        """
        allowed_values = ["MVN", "NPM"]
        if build_type not in allowed_values:
            raise ValueError(
                "Invalid value for `build_type` ({0}), must be one of {1}"
                .format(build_type, allowed_values)
            )

        self._build_type = build_type