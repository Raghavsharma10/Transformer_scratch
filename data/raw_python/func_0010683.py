def _download_file(self, url, apikey=None):
        """ Download lookup files either from Clublog or Country-files.com
        """
        import gzip
        import tempfile

        cty = {}
        cty_date = ""
        cty_file_path = None

        filename = None

        # download file
        if apikey: # clublog
            response = requests.get(url+"?api="+apikey, timeout=10)
        else: # country-files.com
            response = requests.get(url, timeout=10)

        if not self._check_html_response(response):
            raise LookupError

        #Clublog Webserver Header
        if "Content-Disposition" in response.headers:
            f = re.search('filename=".+"', response.headers["Content-Disposition"])
            if f:
                f = f.group(0)
                filename = re.search('".+"', f).group(0).replace('"', '')

        #Country-files.org webserver header
        else:
            f = re.search('/.{4}plist$', url)
            if f:
                f = f.group(0)
                filename = f[1:]

        if not filename:
            filename = "cty_" + self._generate_random_word(5)

        download_file_path = os.path.join(tempfile.gettempdir(), filename)
        with open(download_file_path, "wb") as download_file:
            download_file.write(response.content)
        self._logger.debug(str(download_file_path) + " successfully downloaded")

        # unzip file, if gz
        if os.path.splitext(download_file_path)[1][1:] == "gz":

            download_file = gzip.open(download_file_path, "r")
            try:
                cty_file_path = os.path.join(os.path.splitext(download_file_path)[0])
                with open(cty_file_path, "wb") as cty_file:
                    cty_file.write(download_file.read())
                self._logger.debug(str(cty_file_path) + " successfully extracted")
            finally:
                download_file.close()
        else:
            cty_file_path = download_file_path

        return cty_file_path