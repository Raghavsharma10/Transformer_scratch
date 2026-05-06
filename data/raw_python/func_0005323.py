def _check_filter_specific_md(self, specific_md: list):
        """Check if specific_md parameter is valid.

        :param list specific_md: list of specific metadata UUID to check
        """
        if isinstance(specific_md, list):
            if len(specific_md) > 0:
                # checking UUIDs and poping bad ones
                for md in specific_md:
                    if not self.check_is_uuid(md):
                        specific_md.remove(md)
                        logging.error("Metadata UUID is not correct: {}".format(md))
                # joining survivors
                specific_md = ",".join(specific_md)
            else:
                specific_md = ""
        else:
            raise TypeError("'specific_md' expects a list")
        return specific_md