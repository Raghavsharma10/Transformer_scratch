def train_language_model_from_dataset(self, dataset_id, name, token=None, url=API_TRAIN_LANGUAGE_MODEL):
        """ Trains a model given a dataset and its ID.
            :param dataset_id: string, the ID for a dataset you created previously.
            :param name: string, name for your model.            
            returns: a request object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        dummy_files = {'name': (None, name), 'datasetId':(None, dataset_id)}
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url
        r = requests.post(the_url, headers=h, files=dummy_files)

        return r