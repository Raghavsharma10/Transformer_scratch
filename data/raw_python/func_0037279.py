def _get_player_profile(self, player_handle):
        """Returns pubg player profile from PUBG api, no filtering

            :param player_handle: player PUBG profile name
            :type player_handle: str
            :return: return json from PUBG API
            :rtype: dict
        """
        url = self.pubg_url + player_handle
        response = requests.request("GET", url, headers=self.headers)
        data = json.loads(response.text)
        return data