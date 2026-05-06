def get_language_prediction_from_model(self, model_id, document, token=None, url=API_GET_LANGUAGE_PREDICTION):
        """ Gets a prediction based on a body of text you send to a trained model you created previously.
            :param model_id: string, the ID for a model you created previously.
            :param document: string, a body of text to be classified.
            returns: a request object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        dummy_files = {'modelId': (None, model_id), 'document':(None, document)}
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url
        r = requests.post(the_url, headers=h, files=dummy_files)

        return r