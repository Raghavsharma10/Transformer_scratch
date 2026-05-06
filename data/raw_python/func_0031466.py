def db_import_xml(self, url=None, force_download=False, taxids=None, silent=False):
        """Updates the CTD database
        
        1. downloads gzipped XML
        2. drops all tables in database
        3. creates all tables in database
        4. import XML
        5. close session

        :param Optional[list[int]] taxids: list of NCBI taxonomy identifier
        :param str url: iterable of URL strings
        :param bool force_download: force method to download
        :param bool silent:
        """
        log.info('Update UniProt database from {}'.format(url))

        self._drop_tables()
        xml_gzipped_file_path, version_file_path = self.download(url, force_download)
        self._create_tables()
        self.import_version(version_file_path)
        self.import_xml(xml_gzipped_file_path, taxids, silent)
        self.session.close()