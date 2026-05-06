def upload_image(self, path=None, url=None, title=None, description=None,
                     album=None):
        """
        Upload the image at either path or url.

        :param path: The path to the image you want to upload.
        :param url: The url to the image you want to upload.
        :param title: The title the image will have when uploaded.
        :param description: The description the image will have when uploaded.
        :param album: The album the image will be added to when uploaded. Can
            be either a Album object or it's id. Leave at None to upload
            without adding to an Album, adding it later is possible.
            Authentication as album owner is necessary to upload to an album
            with this function.

        :returns: An Image object representing the uploaded image.
        """
        if bool(path) == bool(url):
            raise LookupError("Either path or url must be given.")
        if path:
            with open(path, 'rb') as image_file:
                binary_data = image_file.read()
                image = b64encode(binary_data)
        else:
            image = url

        payload = {'album_id': album, 'image': image,
                   'title': title, 'description': description}

        resp = self._send_request(self._base_url + "/3/image",
                                  params=payload, method='POST')
        # TEMPORARY HACK:
        # On 5-08-2013 I noticed Imgur now returned enough information from
        # this call to fully populate the Image object. However those variables
        # that matched arguments were always None, even if they had been given.
        # See https://groups.google.com/forum/#!topic/imgur/F3uVb55TMGo
        resp['title'] = title
        resp['description'] = description
        if album is not None:
            resp['album'] = (Album({'id': album}, self, False) if not
                             isinstance(album, Album) else album)
        return Image(resp, self)