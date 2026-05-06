def group_by_source(self):
        """Return a dict of all of the docs, with the source associated
        with the doc as a key"""
        from collections import defaultdict
        docs = defaultdict(list)

        for k, v in self.items():
            if 'source' in v:
                docs[v.source].append(dict(v.items()))

        return docs