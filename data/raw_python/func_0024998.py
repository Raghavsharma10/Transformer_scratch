def curl(self):
        """Sending a single cURL request to download"""
        c = self._pycurl
        # Resume download
        if os.path.exists(self.path) and self.resume:
            mode = 'ab'
            self.downloaded = os.path.getsize(self.path)
            c.setopt(pycurl.RESUME_FROM, self.downloaded)
        else:
            mode = 'wb'
        with open(self.path, mode) as f:
            c.setopt(c.URL, utf8_encode(self.url))
            if self.auth:
                c.setopt(c.USERPWD, '%s:%s' % self.auth)
            c.setopt(c.USERAGENT, self._user_agent)
            c.setopt(c.WRITEDATA, f)
            h = self._get_pycurl_headers()
            if h is not None:
                c.setopt(pycurl.HTTPHEADER, h)
            c.setopt(c.NOPROGRESS, 0)
            c.setopt(pycurl.FOLLOWLOCATION, 1)
            c.setopt(c.PROGRESSFUNCTION, self.progress)
            self._fill_in_cainfo()
            if self._pass_through_opts:
                for key, value in self._pass_through_opts.items():
                    c.setopt(key, value)
            c.perform()