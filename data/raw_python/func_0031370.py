def _dict_factory(cursor, row):
        ''' factory for sqlite3 to return results as dict
        '''
        d = {}
        for idx, col in enumerate(cursor.description):
            if col[0] == 'rowid':
                d['_id'] = row[idx]
            else:
                d[col[0]] = row[idx]
        return d