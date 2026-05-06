def _get_monitoring_heartbeat(self):
        """
        Tests whether or not the ACS service being monitored is alive.
        """
        target = self.uri + '/monitoring/heartbeat'
        response = self.session.get(target)
        return response