def groups_count(self):
        """Number of all groups (get-only).

        :getter: Returns number of all groups
        :type: int
        """
        if self._keyboard_description.contents.ctrls is not None:
            return self._keyboard_description.contents.ctrls.contents.num_groups
        else:
            groups_source = self._groups_source

            groups_count = 0
            while (groups_count < XkbNumKbdGroups and
                   groups_source[groups_count] != None_):
                groups_count += 1

            return groups_count