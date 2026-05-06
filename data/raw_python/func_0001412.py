def read(self):
        ''' Read tagged doc from mutliple files (sents, tokens, concepts, links, tags) '''
        warnings.warn("Document.read() is deprecated and will be removed in near future.", DeprecationWarning)
        with TxtReader.from_doc(self) as reader:
            reader.read(self)
        return self