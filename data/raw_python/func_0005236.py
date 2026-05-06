def download_file_by_key(self, key, dest_dir):
        """Downloads a file by key from the specified S3 bucket

        This method takes the full 'key' as the arg, and attempts to
        download the file to the specified dest_dir as the destination
        directory. This method sets the downloaded filename to be the
        same as it is on S3.

        :param key: (str) S3 key for the file to be downloaded.
        :param dest_dir: (str) Full path destination directory
        :return: (str) Downloaded file destination if the file was
            downloaded successfully, None otherwise.
        """
        log = logging.getLogger(self.cls_logger + '.download_file_by_key')
        if not isinstance(key, basestring):
            log.error('key argument is not a string')
            return None
        if not isinstance(dest_dir, basestring):
            log.error('dest_dir argument is not a string')
            return None
        if not os.path.isdir(dest_dir):
            log.error('Directory not found on file system: %s', dest_dir)
            return None
        try:
            dest_path = self.__download_from_s3(key, dest_dir)
        except S3UtilError:
            raise
        return dest_path