def filter_oids(self, oids):
        '''
        Leaves only objects with specified oids.

        :param oids: list of oids to include
        '''
        oids = set(oids)
        return self[self['_oid'].map(lambda x: x in oids)]