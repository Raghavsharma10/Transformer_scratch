def setup_remoteckan(self, remoteckan=None, **kwargs):
        # type: (Optional[ckanapi.RemoteCKAN], Any) -> None
        """
        Set up remote CKAN from provided CKAN or by creating from configuration

        Args:
            remoteckan (Optional[ckanapi.RemoteCKAN]): CKAN instance. Defaults to setting one up from configuration.

        Returns:
            None

        """
        if remoteckan is None:
            self._remoteckan = self.create_remoteckan(self.get_hdx_site_url(), full_agent=self.get_user_agent(),
                                                      **kwargs)
        else:
            self._remoteckan = remoteckan