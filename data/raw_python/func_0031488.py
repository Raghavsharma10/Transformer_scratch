def download(cls, url=None, force_download=False):
        """Downloads uniprot_sprot.xml.gz and reldate.txt (release date information) from URL or file path

        .. note::

            only URL/path of xml.gz is needed and valid value for parameter url. URL/path for reldate.txt have to be the
            same folder
    
        :param str url: UniProt gzipped URL or file path
        :param force_download: force method to download
        :type force_download: bool
        """
        if url:
            version_url = os.path.join(os.path.dirname(url), defaults.VERSION_FILE_NAME)
        else:
            url = os.path.join(defaults.XML_DIR_NAME, defaults.SWISSPROT_FILE_NAME)
            version_url = os.path.join(defaults.XML_DIR_NAME, defaults.VERSION_FILE_NAME)

        xml_file_path = cls.get_path_to_file_from_url(url)
        version_file_path = cls.get_path_to_file_from_url(version_url)

        if force_download or not os.path.exists(xml_file_path):

            log.info('download {} and {}'.format(xml_file_path, version_file_path))

            scheme = urlsplit(url).scheme

            if scheme in ('ftp', 'http'):
                urlretrieve(version_url, version_file_path)
                urlretrieve(url, xml_file_path)

            elif not scheme and os.path.isfile(url):
                shutil.copyfile(url, xml_file_path)
                shutil.copyfile(version_url, version_file_path)

        return xml_file_path, version_file_path