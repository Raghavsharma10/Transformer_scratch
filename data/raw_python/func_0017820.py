def _current_web_port(self):
        """
        return just the port number for the web container, or None if
        not running
        """
        info = inspect_container(self._get_container_name('web'))
        if info is None:
            return None
        try:
            if not info['State']['Running']:
                return None
            return info['NetworkSettings']['Ports']['5000/tcp'][0]['HostPort']
        except TypeError:
            return None