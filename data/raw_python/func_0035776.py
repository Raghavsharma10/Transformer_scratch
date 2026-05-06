def get_model_info(self, model_id, token=None, url=API_GET_MODEL_INFO):
        """ Gets information about a specific previously trained model, ie: stats and accuracy
            :param model_id: string, model_id previously supplied by the API
            returns: requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url + '/' + model_id
        r = requests.get(the_url, headers=h)

        return r