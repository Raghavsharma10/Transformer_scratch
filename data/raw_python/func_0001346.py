def create_index(self, columns, name=None, index_type="btree"):
        """
        Create an index to speed up queries on a table.
        If no ``name`` is given a random name is created.
        ::
            table.create_index(['name', 'country'])
        """
        self._check_dropped()
        if not name:
            sig = "||".join(columns + [index_type])
            # This is a work-around for a bug in <=0.6.1 which would create
            # indexes based on hash() rather than a proper hash.
            key = abs(hash(sig))
            name = "ix_%s_%s" % (self.table.name, key)
            if name in self.indexes:
                return self.indexes[name]
            key = sha1(sig.encode("utf-8")).hexdigest()[:16]
            name = "ix_%s_%s" % (self.table.name, key)
        if name in self.indexes:
            return self.indexes[name]
        # self.db._acquire()
        columns = [self.table.c[col] for col in columns]
        idx = Index(name, *columns, postgresql_using=index_type)
        idx.create(self.engine)
        # finally:
        #    self.db._release()
        self.indexes[name] = idx
        return idx