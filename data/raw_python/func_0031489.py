def get_path_to_file_from_url(cls, url):
        """standard file path
        
        :param str url: download URL
        """
        file_name = urlparse(url).path.split('/')[-1]
        return os.path.join(PYUNIPROT_DATA_DIR, file_name)