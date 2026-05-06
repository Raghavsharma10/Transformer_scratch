def _thumbnail_s3(self, original_filename, thumb_filename,
                      thumb_size, thumb_url, bucket_name,
                      crop=None, bg=None, quality=85):
        """Finds or creates a thumbnail for the specified image on Amazon S3."""

        scheme = self.app.config.get('THUMBNAIL_S3_USE_HTTPS') and 'https' or 'http'

        thumb_url_full = url_for_s3(
            'static',
            bucket_name=self.app.config.get('THUMBNAIL_S3_BUCKET_NAME'),
            filename=thumb_url,
            scheme=scheme)
        original_url_full = url_for_s3(
            'static',
            bucket_name=bucket_name,
            filename=self._get_s3_path(original_filename).replace('static/', ''),
            scheme=scheme)

        # Return the thumbnail URL now if it already exists on S3.
        # HTTP HEAD request saves us actually downloading the image
        # for this check.
        # Thanks to:
        # http://stackoverflow.com/a/16778749/2066849
        try:
            resp = httplib2.Http().request(thumb_url_full, 'HEAD')
            resp_status = int(resp[0]['status'])
            assert(resp_status < 400)
            return thumb_url_full
        except Exception:
            pass

        # Thanks to:
        # http://stackoverflow.com/a/12020860/2066849
        try:
            fd = urllib.urlopen(original_url_full)
            temp_file = BytesIO(fd.read())
            image = Image.open(temp_file)
        except Exception:
            return ''

        img = self._thumbnail_resize(image, thumb_size, crop=crop, bg=bg)

        temp_file = BytesIO()
        img.save(temp_file, image.format, quality=quality)

        conn = S3Connection(self.app.config.get('THUMBNAIL_S3_ACCESS_KEY_ID'), self.app.config.get('THUMBNAIL_S3_ACCESS_KEY_SECRET'))
        bucket = conn.get_bucket(self.app.config.get('THUMBNAIL_S3_BUCKET_NAME'))

        path = self._get_s3_path(thumb_filename)
        k = bucket.new_key(path)

        try:
            k.set_contents_from_string(temp_file.getvalue())
            k.set_acl(self.app.config.get('THUMBNAIL_S3_ACL', 'public-read'))
        except S3ResponseError:
            return ''

        return thumb_url_full