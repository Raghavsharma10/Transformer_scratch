def _move_path(self):
        """
        Move the downloaded file to the authentic path (identified by
        effective URL)
        """
        if is_temp_path(self._path) and self._pycurl is not None:
            eurl = self._pycurl.getinfo(pycurl.EFFECTIVE_URL)
            er = get_resource_name(eurl)
            r = get_resource_name(self.url)
            if er != r and os.path.exists(self.path):
                new_path = self._get_path(self._path, eurl)
                shutil.move(self.path, new_path)
                self.path = new_path