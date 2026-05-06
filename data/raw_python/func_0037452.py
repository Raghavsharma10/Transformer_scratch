def db_import(self, urls=None, force_download=False):
        """Updates the CTD database
        
        1. downloads all files from CTD
        2. drops all tables in database
        3. creates all tables in database
        4. import all data from CTD files
        
        :param iter[str] urls: An iterable of URL strings
        :param bool force_download: force method to download
        """
        if not urls:
            urls = [
                defaults.url_base + table_conf.tables[model]['file_name']
                for model in table_conf.tables
            ]

        log.info('Update CTD database from %s', urls)

        self.drop_all()
        self.download_urls(urls=urls, force_download=force_download)
        self.create_all()
        self.import_tables()
        self.session.close()