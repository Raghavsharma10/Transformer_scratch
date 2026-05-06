def _write_html(self, filename):
        """Read the html site with the given filename
        from the data directory and write it to :data:`RedirectHandler.wfile`.

        :param filename: the filename to read
        :type filename: :class:`str`
        :returns: None
        :rtype: None
        :raises: None
        """
        datapath = os.path.join('html', filename)
        sitepath = pkg_resources.resource_filename('pytwitcherapi', datapath)
        with open(sitepath, 'r') as f:
            html = f.read()
        self.wfile.write(html.encode('utf-8'))