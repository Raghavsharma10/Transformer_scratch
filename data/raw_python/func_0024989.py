def is_alive(self):
        """
        Will test whether the ACS service is up and alive.
        """
        response = self.get_monitoring_heartbeat()
        if response.status_code == 200 and response.content == 'alive':
            return True

        return False