def create_dataset_synchronous(self, file_url, dataset_type='image', token=None, url=API_CREATE_DATASET):
        """ Creates a dataset so you can train models from it
            :param file_url: string, url to an accessible zip file containing the necessary image files
            and folder structure indicating the labels to train. See docs online.
            :param dataset_type: string, one of the dataset types, available options Nov 2017 were 
            'image', 'image-detection' and 'image-multi-label'.
            returns: requests object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        m = MultipartEncoder(fields={'type':dataset_type, 'path':file_url})
        h = {'Authorization': auth, 'Cache-Control':'no-cache', 'Content-Type':m.content_type}
        the_url = url
        r = requests.post(the_url, headers=h, data=m)

        return r