def _check_filter_specific_tag(self, specific_tag: list):
        """Check if specific_tag parameter is valid.

        :param list specific_tag: list of specific tag to check
        """
        if isinstance(specific_tag, list):
            if len(specific_tag) > 0:
                specific_tag = ",".join(specific_tag)
            else:
                specific_tag = ""
        else:
            raise TypeError("'specific_tag' expects a list")
        return specific_tag