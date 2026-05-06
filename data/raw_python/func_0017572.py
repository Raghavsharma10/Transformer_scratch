def get_player_stats(self, player_key, board_key):
        """
        Calling the Player Stats API
        Args:
            player_key: Key of the player
            board_key: key of the board
        Return:
            json data
        """
        player_stats_url = self.api_path + 'player/' + player_key + '/league/' + board_key + '/stats/'
        response = self.get_response(player_stats_url)
        return response