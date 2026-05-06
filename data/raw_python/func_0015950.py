def get_service_location_info(self, service_location_id):
        """
        Request service location info

        Parameters
        ----------
        service_location_id : int

        Returns
        -------
        dict
        """
        url = urljoin(URLS['servicelocation'], service_location_id, "info")
        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()