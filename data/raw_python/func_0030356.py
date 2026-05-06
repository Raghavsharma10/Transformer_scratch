def content(self, content):
        """
        Sets the content of this SupportLevelPage.

        :param content: The content of this SupportLevelPage.
        :type: list[str]
        """
        allowed_values = ["UNRELEASED", "EARLYACCESS", "SUPPORTED", "EXTENDED_SUPPORT", "EOL"]
        if not set(content).issubset(set(allowed_values)):
            raise ValueError(
                "Invalid values for `content` [{0}], must be a subset of [{1}]"
                .format(", ".join(map(str, set(content)-set(allowed_values))),
                        ", ".join(map(str, allowed_values)))
            )

        self._content = content