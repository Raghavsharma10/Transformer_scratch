def repository_type(self, repository_type):
        """
        Sets the repository_type of this TargetRepositoryRest.

        :param repository_type: The repository_type of this TargetRepositoryRest.
        :type: str
        """
        allowed_values = ["MAVEN", "NPM", "COCOA_POD", "GENERIC_PROXY"]
        if repository_type not in allowed_values:
            raise ValueError(
                "Invalid value for `repository_type` ({0}), must be one of {1}"
                .format(repository_type, allowed_values)
            )

        self._repository_type = repository_type