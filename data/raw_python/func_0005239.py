def find_keys(self, regex, bucket_name=None):
        """Finds a list of S3 keys matching the passed regex

        Given a regular expression, this method searches the S3 bucket
        for matching keys, and returns an array of strings for matched
        keys, an empty array if non are found.

        :param regex: (str) Regular expression to use is the key search
        :param bucket_name: (str) Name of bucket to search (optional)
        :return: Array of strings containing matched S3 keys
        """
        log = logging.getLogger(self.cls_logger + '.find_keys')
        matched_keys = []
        if not isinstance(regex, basestring):
            log.error('regex argument is not a string, found: {t}'.format(t=regex.__class__.__name__))
            return None

        # Determine which bucket to use
        if bucket_name is None:
            s3bucket = self.bucket
        else:
            log.debug('Using the provided S3 bucket: {n}'.format(n=bucket_name))
            s3bucket = self.s3resource.Bucket(bucket_name)

        log.info('Looking up S3 keys based on regex: {r}'.format(r=regex))
        for item in s3bucket.objects.all():
            log.debug('Checking if regex matches key: {k}'.format(k=item.key))
            match = re.search(regex, item.key)
            if match:
                matched_keys.append(item.key)
        log.info('Found matching keys: {k}'.format(k=matched_keys))
        return matched_keys