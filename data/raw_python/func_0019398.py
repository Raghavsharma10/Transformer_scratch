def get_game_id(self, date):
        """Returns the Game ID associated with the date that is passed in.

        Parameters
        ----------
        date : str
            The date associated with the game whose Game ID. The date that is
            passed in can take on a numeric format of MM/DD/YY (like "01/06/16"
            or "01/06/2016") or the expanded Month Day, Year format (like
            "Jan 06, 2016" or "January 06, 2016").

        Returns
        -------
        game_id : str
            The desired Game ID.
        """
        df = self.get_game_logs()
        game_id = df[df.GAME_DATE == date].Game_ID.values[0]
        return game_id