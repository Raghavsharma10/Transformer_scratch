def thumbnail(self, img_url, size, crop=None, bg=None, quality=85,
                  storage_type=None, bucket_name=None):
        """
        :param img_url: url img - '/assets/media/summer.jpg'
        :param size: size return thumb - '100x100'
        :param crop: crop return thumb - 'fit' or None
        :param bg: tuple color or None - (255, 255, 255, 0)
        :param quality: JPEG quality 1-100
        :param storage_type: either 's3' or None
        :param bucket_name: s3 bucket name
        :return: :thumb_url:
        """

        width, height = [int(x) for x in size.split('x')]
        thumb_size = (width, height)
        url_path, img_name = os.path.split(img_url)
        name, fm = os.path.splitext(img_name)

        miniature = self._get_name(name, fm, size, crop, bg, quality)

        original_filename = os.path.join(self.app.config['MEDIA_FOLDER'], url_path, img_name)
        thumb_filename = os.path.join(self.app.config['MEDIA_THUMBNAIL_FOLDER'], url_path, miniature)

        thumb_url = os.path.join(self.app.config['MEDIA_THUMBNAIL_URL'], url_path, miniature)

        if not (storage_type and bucket_name):
            return self._thumbnail_local(original_filename,
                                         thumb_filename,
                                         thumb_size,
                                         thumb_url,
                                         crop=crop,
                                         bg=bg,
                                         quality=quality)
        else:
            if storage_type != 's3':
                raise ValueError('Storage type "%s" is invalid, the only supported storage type (apart from default local storage) is s3.' % storage_type)

            return self._thumbnail_s3(original_filename,
                                      thumb_filename,
                                      thumb_size,
                                      thumb_url,
                                      bucket_name,
                                      crop=crop,
                                      bg=bg,
                                      quality=quality)