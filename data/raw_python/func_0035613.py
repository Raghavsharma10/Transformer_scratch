def upload(self, image, name=None):
        """ Upload the given image, which can be a http[s] URL, a path to an existing file,
            binary image data, or an open file handle.
        """
        assert self.client_id, "imgur client ID is not set! Export the IMGUR_CLIENT_ID environment variable..."
        assert self.client_secret, "imgur client secret is not set! Export the IMGUR_CLIENT_SECRET environment variable..."

        # Prepare image
        try:
            image_data = (image + '')
        except (TypeError, ValueError):
            assert hasattr(image, "read"), "Image is neither a string nor an open file handle"
            image_type = "file"
            image_data = image  # XXX are streams supported? need a temp file?
            image_repr = repr(image)
        else:
            if image.startswith("http:") or image.startswith("https:"):
                image_type = "url"
                image_data = image
                image_repr = image
            elif all(ord(i) >= 32 for i in image) and os.path.exists(image):
                image_type = "file"
                image_data = image  # XXX open(image, "rb")
                image_repr = "file:" + image
            else:
                # XXX Not supported anymore (maybe use a temp file?)
                image_type = "base64"
                image_data = image_data.encode(image_type)
                image_repr = "<binary data>"

        # Upload image
        # XXX "name",    name or hashlib.md5(str(image)).hexdigest()),
        client = ImgurClient(self.client_id, self.client_secret)
        result = (client.upload_from_url if image_type == 'url'
                  else client.upload_from_path)(image_data)  # XXX config=None, anon=True)

        if result['link'].startswith('http:'):
            result['link'] = 'https:' + result['link'][5:]
        result['hash'] = result['id']  # compatibility to API v1
        result['caption'] = result['description']  # compatibility to API v1

        return parts.Bunch(
            image=parts.Bunch(result),
            links=parts.Bunch(
                delete_page=None,
                imgur_page=None,
                original=result['link'],
                large_thumbnail="{0}s.{1}".format(*result['link'].rsplit('.', 1)),
                small_square="{0}l.{1}".format(*result['link'].rsplit('.', 1)),
            ))