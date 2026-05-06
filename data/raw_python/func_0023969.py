def _save_token_cache(self, new_cache):
        'Write out to the filesystem a cache of the OAuth2 information.'
        logging.debug('Looking to write to local authentication cache...')
        if not self._check_token_cache_type(new_cache):
            logging.error('Attempt to save a bad value: %s', new_cache)
            return
        try:
            logging.debug('About to write to fs cache file: %s',
                          self.token_cache_file)
            with open(self.token_cache_file, 'wb') as f:
                cPickle.dump(new_cache, f, protocol=cPickle.HIGHEST_PROTOCOL)
                logging.debug('Finished dumping cache_value to fs cache file.')
        except:
            logging.exception(
                'Could not successfully cache OAuth2 secrets on the file '
                'system.')