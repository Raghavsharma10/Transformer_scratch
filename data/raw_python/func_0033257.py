def _setup_preferred_paths(self, preferred_conversion_paths):
        '''
        Add given valid preferred conversion paths
        '''
        for path in preferred_conversion_paths:
            for pair in pair_looper(path):
                if pair not in self.converters:
                    log.warning('Invalid conversion path %s, unknown step %s' %
                                (repr(path), repr(pair)))
                    break
            else:
                # If it did not break, then add to dgraph
                self.dgraph.add_preferred_path(*path)