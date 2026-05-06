def autoschema(self, objects, **kwargs):
        ''' wrapper around utils.autoschema function '''
        return autoschema(objects=objects, exclude_keys=self.RESTRICTED_KEYS,
                          **kwargs)