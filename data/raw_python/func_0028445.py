def _heartbeat_view(self):
        """
        Runs all the registered checks and returns a JSON response with either
        a status code of 200 or 500 depending on the results of the checks.

        Any check that returns a warning or worse (error, critical) will
        return a 500 response.
        """
        details = {}
        statuses = {}
        level = 0

        for name, check in self.checks.items():
            detail = self._heartbeat_check_detail(check)
            statuses[name] = detail['status']
            level = max(level, detail['level'])
            if detail['level'] > 0:
                details[name] = detail

        payload = {
            'status': checks.level_to_text(level),
            'checks': statuses,
            'details': details,
        }

        def render(status_code):
            return make_response(jsonify(payload), status_code)

        if level < checks.WARNING:
            status_code = 200
            heartbeat_passed.send(self, level=level)
            return render(status_code)
        else:
            status_code = 500
            heartbeat_failed.send(self, level=level)
            raise HeartbeatFailure(response=render(status_code))