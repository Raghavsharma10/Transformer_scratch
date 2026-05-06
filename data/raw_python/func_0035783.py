def get_training_status(self, model_id, token=None, url=API_TRAIN_MODEL):
        """ Gets status on the training process once you create a model
            :param model_id: string, id of the model to check
            returns: requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url + '/' + model_id
        r = requests.get(the_url, headers=h)

        return r