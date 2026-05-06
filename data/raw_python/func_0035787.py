def get_language_model_status(self, model_id, token=None, url=API_TRAIN_LANGUAGE_MODEL):
        """ Gets the status of your model, including whether the training has finished.
            :param model_id: string, the ID for a model you created previously.            
            returns: a request object
        """
        auth = 'Bearer ' + self.check_for_token(token)        
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url + '/' + model_id
        r = requests.get(the_url, headers=h)

        return r