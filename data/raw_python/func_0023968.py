def _load_token_cache(self):
        'Reads the local fs cache for pre-authorized access tokens'
        try:
            logging.debug('About to read from local file cache file %s',
                          self.token_cache_file)
            with open(self.token_cache_file, 'rb') as f:
                fs_cached = cPickle.load(f)
                if self._check_token_cache_type(fs_cached):
                    logging.debug('Loaded from file system: %s', fs_cached)
                    return fs_cached
                else:
                    logging.warn('Found unexpected value in cache. %s',
                                 fs_cached)
                    return None
        except IOError:
            logging.debug(
                'Did not find file: %s on the file system.',
                self.token_cache_file)
            return None
        except:
            logging.info(
                'Encountered exception loading from the file system.',
                exc_info=True)
            return None