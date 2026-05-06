def add(self, src):
        """
        :param src: file path
        :return: checksum value
        """

        checksum = get_checksum(src)

        filename = self.get_filename(checksum)

        if not filename:

            new_name = self._get_new_name()
            new_realpath = self._storage_dir + '/' + new_name

            os.makedirs(os.path.split(new_realpath)[0], exist_ok=True)

            shutil.copyfile(src, new_realpath)

            self._log[new_name] = {
                'checksum': checksum,
                'mtime': os.path.getmtime(new_realpath),
                'size': os.path.getsize(new_realpath)
            }

            self.write_log()
        return checksum