def request(self, *args, **kwargs):
        """
        The main purpose of this is to be a wrapper-like function to pass the api_token and all the other params to the
        requests that are being made

        :returns: An instance of RequestsHandler
        """
        return RequestsHandler(*args, api_token=self.api_token,
                               verify=self.mist_client.verify,
                               job_id=self.mist_client.job_id, **kwargs)