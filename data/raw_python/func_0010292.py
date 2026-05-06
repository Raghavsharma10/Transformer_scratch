def file_system_service(self):
        """Property providing access to the :class:`.FileSystemServiceAPI`"""
        if self._fss_api is None:
            self._fss_api = self.get_fss_api()
        return self._fss_api