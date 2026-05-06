def create_language_dataset_from_url(self, file_url, token=None, url=API_CREATE_LANGUAGE_DATASET):
        """ Creates a dataset from a publicly accessible file stored in the cloud.
            :param file_url: string, in the form of a URL to a file accessible on the cloud. 
            Popular options include Dropbox, AWS S3, Google Drive.
            warning: Google Drive by default gives you a link to a web ui that allows you to download a file
            NOT to the file directly. There is a way to change the link to point directly to the file as of 2018
            as this may change, please search google for a solution.
            returns: a request object
        """
        auth = 'Bearer ' + self.check_for_token(token)
        dummy_files = {'type': (None, 'text-intent'), 'path':(None, file_url)}
        h = {'Authorization': auth, 'Cache-Control':'no-cache'}
        the_url = url
        r = requests.post(the_url, headers=h, files=dummy_files)

        return r