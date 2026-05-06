def get_datasets_info(self, token=None, url=API_GET_DATASETS_INFO):
        """ Gets information on all datasets for this account
            returns: requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url
        r = requests.get(the_url, headers=h)

        return r