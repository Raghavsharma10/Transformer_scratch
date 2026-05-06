def find_key(self, regex):
        """Attempts to find a single S3 key based on the passed regex

        Given a regular expression, this method searches the S3 bucket
        for a matching key, and returns it if exactly 1 key matches.
        Otherwise, None is returned.

        :param regex: (str) Regular expression for an S3 key
        :return: (str) Full length S3 key matching the regex, None
            otherwise
        """
        log = logging.getLogger(self.cls_logger + '.find_key')
        if not isinstance(regex, basestring):
            log.error('regex argument is not a string')
            return None
        log.info('Looking up a single S3 key based on regex: %s', regex)
        matched_keys = []
        for item in self.bucket.objects.all():
            log.debug('Checking if regex matches key: %s', item.key)
            match = re.search(regex, item.key)
            if match:
                matched_keys.append(item.key)
        if len(matched_keys) == 1:
            log.info('Found matching key: %s', matched_keys[0])
            return matched_keys[0]
        elif len(matched_keys) > 1:
            log.info('Passed regex matched more than 1 key: %s', regex)
            return None
        else:
            log.info('Passed regex did not match any key: %s', regex)
            return None