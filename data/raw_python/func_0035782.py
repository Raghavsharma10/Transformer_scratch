def train_model(self, dataset_id, model_name, token=None, url=API_TRAIN_MODEL):
        """ Train a model given a specifi dataset previously created
            :param dataset_id: string, the id of a previously created dataset
            :param model_name: string, what you will call this model
            attention: This may take a while and a response will be returned before the model has
            finished being trained. See docos and method get_training_status.
            returns: requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        m = MultipartEncoder(fields={'name':model_name, 'datasetId':dataset_id})
        h = {'Authorization': auth, 'Cache-Control':'no-cache', 'Content-Type':m.content_type}
        the_url = url
        r = requests.post(the_url, headers=h, data=m)

        return r