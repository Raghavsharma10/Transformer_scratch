def nested_assign(self, key_list, value):
        """ Set the value of nested LIVVDicts given a list """
        if len(key_list) == 1:
            self[key_list[0]] = value
        elif len(key_list) > 1:
            if key_list[0] not in self:
                self[key_list[0]] = LIVVDict()
            self[key_list[0]].nested_assign(key_list[1:], value)