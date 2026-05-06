def _find_bad_meta(self):
        '''Fill self._badmeta with meta datatypes that are invalid'''
        self._badmeta = dict()

        for datatype in self.meta:
            for item in self.meta[datatype]:
                if not Dap._meta_valid[datatype].match(item):
                    if datatype not in self._badmeta:
                        self._badmeta[datatype] = []
                    self._badmeta[datatype].append(item)