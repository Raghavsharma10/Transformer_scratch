def get_b64_image_prediction(self, model_id, b64_encoded_string, token=None, url=API_GET_PREDICTION_IMAGE_URL):
        """ Gets a prediction from a supplied image enconded as a b64 string, useful when uploading
            images to a server backed by this library.
            :param model_id: string, once you train a model you'll be given a model id to use.
            :param b64_encoded_string: string, a b64 enconded string representation of an image.
            returns: requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)        
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url
    
        encoded_string = b64_encoded_string

        m = MultipartEncoder(fields={'sampleBase64Content':encoded_string, 'modelId':model_id})
        h = {'Authorization': auth, 'Cache-Control':'no-cache', 'Content-Type':m.content_type}
        r = requests.post(the_url, headers=h, data=m)

        return r