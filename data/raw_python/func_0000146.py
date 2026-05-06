def upload(self, release, filename, content_type=None):
        """Upload a file to a release

        :param filename: filename to upload
        :param content_type: optional content type
        :return: json object from github
        """
        release = self.as_id(release)
        name = os.path.basename(filename)
        if not content_type:
            content_type, _ = mimetypes.guess_type(name)
        if not content_type:
            raise ValueError('content_type not known')
        inputs = {'name': name}
        url = '%s%s/%s/assets' % (self.uploads_url,
                                  urlsplit(self.api_url).path,
                                  release)
        info = os.stat(filename)
        size = info[stat.ST_SIZE]
        response = self.http.post(
            url, data=stream_upload(filename), auth=self.auth,
            params=inputs,
            headers={'content-type': content_type,
                     'content-length': str(size)})
        response.raise_for_status()
        return response.json()