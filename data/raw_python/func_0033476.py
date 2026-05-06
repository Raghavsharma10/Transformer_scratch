def deptree(self, field, oids, date=None, level=None, table=None):
        '''
        Dependency tree builder. Recursively fetchs objects that
        are children of the initial set of parent object ids provided.

        :param field: Field that contains the 'parent of' data
        :param oids: Object oids to build depedency tree for
        :param date: date (metrique date range) that should be queried.
                    If date==None then the most recent versions of the
                    objects will be queried.
        :param level: limit depth of recursion
        '''
        table = self.get_table(table)
        fringe = str2list(oids)
        checked = set(fringe)
        loop_k = 0
        while len(fringe) > 0:
            if level and loop_k == abs(level):
                break
            query = '_oid in %s' % list(fringe)
            docs = self.find(table=table, query=query, fields=[field],
                             date=date, raw=True)
            fringe = {oid for doc in docs for oid in (doc[field] or [])
                      if oid not in checked}
            checked |= fringe
            loop_k += 1
        return sorted(checked)