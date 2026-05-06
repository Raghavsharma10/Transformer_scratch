def finalize_structure(self):
        """Any functions needed to cleanup the structure."""
        self.group_list.append(self.current_group)
        group_set = get_unique_groups(self.group_list)
        for item in self.group_list:
            self.group_type_list.append(group_set.index(item))
        self.group_list = [x.convert_to_dict() for x in group_set]