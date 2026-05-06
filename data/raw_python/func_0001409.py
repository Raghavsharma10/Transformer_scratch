def add_sent(self, sent_obj):
        ''' Add a ttl.Sentence object to this document '''
        if sent_obj is None:
            raise Exception("Sentence object cannot be None")
        elif sent_obj.ID is None:
            # if sentID is None, create a new ID
            sent_obj.ID = next(self.__idgen)
        elif self.has_id(sent_obj.ID):
            raise Exception("Sentence ID {} exists".format(sent_obj.ID))
        self.__sent_map[sent_obj.ID] = sent_obj
        self.__sents.append(sent_obj)
        return sent_obj