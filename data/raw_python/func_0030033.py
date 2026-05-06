def artifact_quality(self, artifact_quality):
        """
        Sets the artifact_quality of this ArtifactRest.

        :param artifact_quality: The artifact_quality of this ArtifactRest.
        :type: str
        """
        allowed_values = ["NEW", "VERIFIED", "TESTED", "DEPRECATED", "BLACKLISTED", "DELETED", "TEMPORARY"]
        if artifact_quality not in allowed_values:
            raise ValueError(
                "Invalid value for `artifact_quality` ({0}), must be one of {1}"
                .format(artifact_quality, allowed_values)
            )

        self._artifact_quality = artifact_quality