def incver(self):
        """Increment all of the version numbers"""
        d = {}
        for p in self.__mapper__.attrs:
            if p.key in ['vid','vname','fqname', 'version', 'cache_key']:
                continue
            if p.key == 'revision':
                d[p.key] = self.revision + 1
            else:
                d[p.key] = getattr(self, p.key)

        n =  Dataset(**d)

        return n