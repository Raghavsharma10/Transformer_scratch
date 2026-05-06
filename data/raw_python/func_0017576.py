def get_schedule(self, date=None):
        """
        Calling the Schedule API.

        Return:
           json data
        """

        schedule_url = self.api_path + "schedule/"
        params = {}
        if date:
            params['date'] = date
        response = self.get_response(schedule_url, params)
        return response