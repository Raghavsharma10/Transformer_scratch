def get_season_team(self, season_key, season_team_key,stats_type=None):
        """
        Calling Season teams API

        Arg:
            season_key: key of the season
        Return:
            json data
        """
        params = {"stats_type": stats_type}
        season_team_url = self.api_path + 'season/' +  season_key + '/team/' + season_team_key + '/'
        response = self.get_response(season_team_url, params=params)
        return response