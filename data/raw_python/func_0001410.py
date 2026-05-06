def new_sent(self, text, ID=None, **kwargs):
        ''' Create a new sentence and add it to this Document '''
        if ID is None:
            ID = next(self.__idgen)
        return self.add_sent(Sentence(text, ID=ID, **kwargs))