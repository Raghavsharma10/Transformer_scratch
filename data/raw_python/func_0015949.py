def get_service_locations(self):
        """
        Request service locations

        Returns
        -------
        dict
        """
        url = URLS['servicelocation']
        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()