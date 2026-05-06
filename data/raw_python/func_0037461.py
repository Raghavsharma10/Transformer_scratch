def download_urls(cls, urls, force_download=False):
        """Downloads all CTD URLs that don't exist
    
        :param iter[str] urls: iterable of URL of CTD
        :param bool force_download: force method to download
        """
        for url in urls:
            file_path = cls.get_path_to_file_from_url(url)

            if os.path.exists(file_path) and not force_download:
                log.info('already downloaded %s to %s', url, file_path)
            else:
                log.info('downloading %s to %s', url, file_path)
                download_timer = time.time()
                urlretrieve(url, file_path)
                log.info('downloaded in %.2f seconds', time.time() - download_timer)