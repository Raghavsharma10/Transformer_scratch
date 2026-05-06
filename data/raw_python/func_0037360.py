def update(cls, spec, updates, upsert=False):
        '''
        The spec is used to search for the data to update, updates contains the
        values to be updated, and upsert specifies whether to do an insert if
        the original data is not found.
        '''
        if 'key' in spec:
            previous = cls.get(spec['key'])
        else:
            previous = None
        if previous:
            # Update existing data.
            current = cls(**previous.__dict__)
        elif upsert:
            # Create new data.
            current = cls(**spec)
        else:
            current = None
        # XXX Should there be any error thrown if this is a noop?
        if current:
            current.__dict__.update(updates)
            current.save()
        return current