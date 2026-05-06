def get_bundle(self, bundle_name, extensions=None):
        """ Get all the chunks contained in a bundle """
        if self.stats.get('status') == 'done':
            bundle = self.stats.get('chunks', {}).get(bundle_name, None)
            if bundle is None:
                raise KeyError('No such bundle {0!r}.'.format(bundle_name))
            test = self._chunk_filter(extensions)
            return [self._add_url(c) for c in bundle if test(c)]
        elif self.stats.get('status') == 'error':
            raise RuntimeError("{error}: {message}".format(**self.stats))
        else:
            raise RuntimeError(
                "Bad webpack stats file {0} status: {1!r}"
                .format(self.state.stats_file, self.stats.get('status')))