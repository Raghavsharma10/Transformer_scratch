def update_params(self, parameters):
        """Pass in a dictionary to update url parameters for NBA stats API

        Parameters
        ----------
        parameters : dict
            A dict containing key, value pairs that correspond with NBA stats
            API parameters.

        Returns
        -------
        self : TeamLog
            The TeamLog object containing the updated NBA stats API
            parameters.
        """
        self.url_paramaters.update(parameters)
        self.response = requests.get(self.base_url, params=self.url_paramaters,
                                     headers=HEADERS)
        # raise error if status code is not 200
        self.response.raise_for_status()
        return self