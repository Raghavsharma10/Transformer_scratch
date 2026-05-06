def validate(self, sources):
        """Validate the format of sources
        """
        if not isinstance(sources, Root):
            raise Exception("Source object expected")

        parameters = self.get_uri_with_missing_parameters(sources)
        for parameter in parameters:
            logging.getLogger().warn('Missing parameter "%s" in uri of method "%s" in versions "%s"' % (parameter["name"], parameter["method"], parameter["version"]))