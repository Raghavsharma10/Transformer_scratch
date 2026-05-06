def group_variant(self):
        """Current group variant (get-only).

        :getter: Returns current group variant
        :type: str
        """
        v_mapping = {symdata.index: symdata.variant for symdata in self._symboldata_list}
        return v_mapping[self.group_num] or ""