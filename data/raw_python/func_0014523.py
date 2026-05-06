def disable_directory_service(self, check_peer=False):
        """Disable the directory service.

        :param check_peer: If True, disables server authenticity
                           enforcement. If False, disables directory
                           service integration.
        :type check_peer: bool, optional

        :returns: A dictionary describing the status of the directory service.
        :rtype: ResponseDict

        """
        if check_peer:
            return self.set_directory_service(check_peer=False)
        return self.set_directory_service(enabled=False)