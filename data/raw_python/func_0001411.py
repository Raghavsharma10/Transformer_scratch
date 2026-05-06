def pop(self, sent_id, **kwargs):
        ''' If sent_id exists, remove and return the associated sentence object else return default.
        If no default is provided, KeyError will be raised.'''
        if sent_id is not None and not isinstance(sent_id, int):
            sent_id = int(sent_id)
        if not self.has_id(sent_id):
            if 'default' in kwargs:
                return kwargs['default']
            else:
                raise KeyError("Sentence ID {} does not exist".format(sent_id))
        else:
            # sentence exists ...
            sent_obj = self.get(sent_id)
            self.__sent_map.pop(sent_id)
            self.__sents.remove(sent_obj)
            return sent_obj