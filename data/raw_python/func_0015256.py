def _isvalid(self, datatype):
        '''Checks if the given datatype is valid in meta'''
        if datatype in self.meta:
            return bool(Dap._meta_valid[datatype].match(self.meta[datatype]))
        else:
            return datatype in Dap._optional_meta