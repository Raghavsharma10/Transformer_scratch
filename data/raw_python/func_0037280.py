def player(self, player_handle):
        """Returns the full set of data on a player, no filtering"""
        try:
            url = self.pubg_url + player_handle
            response = requests.request("GET", url, headers=self.headers)
            return json.loads(response.text)
        except BaseException as error:
            print('Unhandled exception: ' + str(error))
            raise