def nested_insert(self, item_list):
        """ Create a series of nested LIVVDicts given a list """
        if len(item_list) == 1:
            self[item_list[0]] = LIVVDict()
        elif len(item_list) > 1:
            if item_list[0] not in self:
                self[item_list[0]] = LIVVDict()
            self[item_list[0]].nested_insert(item_list[1:])