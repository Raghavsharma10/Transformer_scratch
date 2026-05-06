def get_season_schedule(self, season_key):
        """
        Calling specific season schedule

        Arg:
           season_key: key of the season
        Return:
           json data
        """

        schedule_url = self.api_path + "season/" + season_key + "/schedule/"
        response = self.get_response(schedule_url)
        return response