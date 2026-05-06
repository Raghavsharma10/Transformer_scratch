def get_models_info_for_dataset(self, dataset_id, token=None, url=API_GET_MODELS):
        """ Gets metadata on all models available for given dataset id
            :param dataset_id: string, previously obtained dataset id
            warning: if providing your own url here, also include the dataset_id in the right place
            as this method will not include it for you. Otherwise use the dataset_id attribute as 
            per usual
            returns: a requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        if url != API_GET_MODELS:
            r = requests.get(the_url, headers=h)
            return r

        the_url = url.replace('<dataset_id>', dataset_id)
        r = requests.get(the_url, headers=h)

        return r