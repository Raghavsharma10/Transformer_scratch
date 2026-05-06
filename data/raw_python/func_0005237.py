def download_file(self, regex, dest_dir):
        """Downloads a file by regex from the specified S3 bucket

        This method takes a regular expression as the arg, and attempts
        to download the file to the specified dest_dir as the
        destination directory. This method sets the downloaded filename
        to be the same as it is on S3.

        :param regex: (str) Regular expression matching the S3 key for
            the file to be downloaded.
        :param dest_dir: (str) Full path destination directory
        :return: (str) Downloaded file destination if the file was
            downloaded successfully, None otherwise.
        """
        log = logging.getLogger(self.cls_logger + '.download_file')
        if not isinstance(regex, basestring):
            log.error('regex argument is not a string')
            return None
        if not isinstance(dest_dir, basestring):
            log.error('dest_dir argument is not a string')
            return None
        if not os.path.isdir(dest_dir):
            log.error('Directory not found on file system: %s', dest_dir)
            return None
        key = self.find_key(regex)
        if key is None:
            log.warn('Could not find a matching S3 key for: %s', regex)
            return None
        return self.__download_from_s3(key, dest_dir)