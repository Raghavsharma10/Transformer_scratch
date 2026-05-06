def index_list(self):
        '''
        List all cube indexes

        :param collection: cube name
        :param owner: username of cube owner
        '''
        logger.info('Listing indexes')
        _ix = {}
        _i = self.inspector
        for tbl in _i.get_table_names():
            _ix.setdefault(tbl, [])
            for ix in _i.get_indexes(tbl):
                _ix[tbl].append(ix)
        return _ix