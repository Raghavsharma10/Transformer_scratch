def download(self, path='', name=None, overwrite=False, size=None):
        """
        Download the image.

        :param path: The image will be downloaded to the folder specified at
            path, if path is None (default) then the current working directory
            will be used.
        :param name: The name the image will be stored as (not including file
            extension). If name is None, then the title of the image will be
            used. If the image doesn't have a title, it's id will be used. Note
            that if the name given by name or title is an invalid filename,
            then the hash will be used as the name instead.
        :param overwrite: If True overwrite already existing file with the same
            name as what we want to save the file as.
        :param size: Instead of downloading the image in it's original size, we
            can choose to instead download a thumbnail of it. Options are
            'small_square', 'big_square', 'small_thumbnail',
            'medium_thumbnail', 'large_thumbnail' or 'huge_thumbnail'.

        :returns: Name of the new file.
        """
        def save_as(filename):
            local_path = os.path.join(path, filename)
            if os.path.exists(local_path) and not overwrite:
                raise Exception("Trying to save as {0}, but file "
                                "already exists.".format(local_path))
            with open(local_path, 'wb') as out_file:
                out_file.write(resp.content)
            return local_path
        valid_sizes = {'small_square': 's', 'big_square': 'b',
                       'small_thumbnail': 't', 'medium_thumbnail': 'm',
                       'large_thumbnail': 'l', 'huge_thumbnail': 'h'}
        if size is not None:
            size = size.lower().replace(' ', '_')
            if size not in valid_sizes:
                raise LookupError('Invalid size. Valid options are: {0}'.format(
                                  ", " .join(valid_sizes.keys())))
        suffix = valid_sizes.get(size, '')
        base, sep, ext = self.link.rpartition('.')
        resp = requests.get(base + suffix + sep + ext)
        if name or self.title:
            try:
                return save_as((name or self.title) + suffix + sep + ext)
            except IOError:
                pass
            # Invalid filename
        return save_as(self.id + suffix + sep + ext)