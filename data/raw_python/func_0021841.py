def fetch_uri(self, directory, uri):
        """
        Use ``urllib.urlretrieve`` to download package to file in sandbox dir.

        @param directory: directory to download to
        @type directory: string

        @param uri: uri to download
        @type uri: string

        @returns: 0 = success or 1 for failed download
        """
        filename = os.path.basename(urlparse(uri)[2])
        if os.path.exists(filename):
            self.logger.error("ERROR: File exists: " + filename)
            return 1

        try:
            downloaded_filename, headers = urlretrieve(uri, filename)
            self.logger.info("Downloaded ./" + filename)
        except IOError as err_msg:
            self.logger.error("Error downloading package %s from URL %s"  \
                    % (filename, uri))
            self.logger.error(str(err_msg))
            return 1

        if headers.gettype() in ["text/html"]:
            dfile = open(downloaded_filename)
            if re.search("404 Not Found", "".join(dfile.readlines())):
                dfile.close()
                self.logger.error("'404 Not Found' error")
                return 1
            dfile.close()
        return 0