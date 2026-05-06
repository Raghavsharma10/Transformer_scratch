def build(self, docs=None, filename=None):
        """Build FM-index
        Params:
            <iterator> | <generator> docs
            <str> filename
        """
        if docs:
            if hasattr(docs, 'items'):
                for (idx, doc) in sorted(getattr(docs, 'items')(),
                                         key=lambda x: x[0]):
                    self.fm.push_back(doc)
            else:
                for doc in filter(bool, docs):
                    self.fm.push_back(doc)
        self.fm.build()
        if filename:
            self.fm.write(filename)