def stop_ckan(self):
        """
        Stop and remove the web container
        """
        remove_container(self._get_container_name('web'), force=True)
        remove_container(self._get_container_name('datapusher'), force=True)