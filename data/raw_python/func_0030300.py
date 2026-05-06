def support_level(self, support_level):
        """
        Sets the support_level of this ProductReleaseRest.

        :param support_level: The support_level of this ProductReleaseRest.
        :type: str
        """
        allowed_values = ["UNRELEASED", "EARLYACCESS", "SUPPORTED", "EXTENDED_SUPPORT", "EOL"]
        if support_level not in allowed_values:
            raise ValueError(
                "Invalid value for `support_level` ({0}), must be one of {1}"
                .format(support_level, allowed_values)
            )

        self._support_level = support_level