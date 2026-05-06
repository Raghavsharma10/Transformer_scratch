def _set_alarm(self, status, home_id):
        """Set alarm satus."""
        response = self._request(
            MINUT_HOMES_URL + "/{}".format(home_id),
            request_type='PUT',
            json={'alarm_status': status})
        return response.get('alarm_status', '') == status