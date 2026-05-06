def get_url_image_prediction(self, model_id, picture_url, token=None, url=API_GET_PREDICTION_IMAGE_URL):
        """ Gets a prediction from a supplied picture url based on a previously trained model.
            :param model_id: string, once you train a model you'll be given a model id to use.
            :param picture_url: string, in the form of a url pointing to a publicly accessible
            image file.
            returns: requests object 
        """
        auth = 'Bearer ' + self.check_for_token(token)
        m = MultipartEncoder(fields={'sampleLocation':picture_url, 'modelId':model_id})
        h = {'Authorization': auth, 'Cache-Control':'no-cache', 'Content-Type':m.content_type}
        the_url = url
        r = requests.post(the_url, headers=h, data=m)

        return r