def _add_variants(self, key, value, schema):
        ''' also possible to define some function that takes
            current value and creates a new value from it
        '''
        variants = schema.get('variants')
        obj = {}
        if variants:
            for _key, func in variants.iteritems():
                _value = func(value, self.store)
                obj.update({_key: _value})
        return obj