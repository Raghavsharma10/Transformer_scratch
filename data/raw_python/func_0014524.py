def enable_directory_service(self, check_peer=False):
        """Enable the directory service.

        :param check_peer: If True, enables server authenticity
                           enforcement. If False, enables directory
                           service integration.
        :type check_peer: bool, optional

        :returns: A dictionary describing the status of the directory service.
        :rtype: ResponseDict

        """
        if check_peer:
            return self.set_directory_service(check_peer=True)
        return self.set_directory_service(enabled=True)