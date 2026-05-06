def strings(self):
        """
        The structure of the bar graph. A list that contains all the strings
        that build up a bar in the given position. thise strings are inside a
        list, starting from the top level.

        structure: [
            1st bar -> ['1st level str', ..., 'height-1 str', 'height str'],
            2nd bar -> ['1st level str', ..., 'height-1 str', 'height str'],
           ...
            last bar -> ...
        ]
        """

        def get_strings(stack_id, height):
            def _str(i):
                if i == None:
                    return VOID_STR
                return self.__list[i]

            strings = list()
            for level in range(1, height + 1):
                _len = len(self.__list) * level
                if _len > stack_id:
                    idx = stack_id - _len
                    if (-1 * idx > len(self.__list)):
                        idx = None
                elif stack_id >= _len:
                    idx = -1
                else:
                    idx = _len - stack_id
                _s = _str(idx)
                strings.append(_s)
            return strings

        has_0 = min(self.data) == 0
        self.__list = STACK_VALUES
        if has_0:
            self.__list = STACK_VALUES_0
        mapped_values = ([self.__get_stack_id(x, self.data, self.height)
                                                        for x in self.data])
        return ([get_strings(stack_id, self.height)
                        for stack_id in mapped_values])