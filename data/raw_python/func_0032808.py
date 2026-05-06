def group_symbol(self):
        """Current group symbol.

        :getter: Returns current group symbol
        :setter: Sets current group symbol
        :type: str
        """
        s_mapping = {symdata.index: symdata.symbol for symdata in self._symboldata_list}
        return s_mapping[self.group_num]