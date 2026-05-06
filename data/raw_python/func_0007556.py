def __set_version_id(self):
        """
        Pulles the versioning info for the request from the child request.
        """

        version_id = self.client.factory.create('VersionId')
        version_id.ServiceId = self._version_info['service_id']
        version_id.Major = self._version_info['major']
        version_id.Intermediate = self._version_info['intermediate']
        version_id.Minor = self._version_info['minor']
        self.logger.debug(version_id)
        self.VersionId = version_id