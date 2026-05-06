def _strip_metadata(self, my_dict):
        """
        Create a copy of dict and remove not needed data
        """
        new_dict = copy.deepcopy(my_dict)
        if const.START in new_dict:
            del new_dict[const.START]
        if const.END in new_dict:
            del new_dict[const.END]
        if const.WHITELIST in new_dict:
            del new_dict[const.WHITELIST]
        if const.WHITELIST_START in new_dict:
            del new_dict[const.WHITELIST_START]
        if const.WHITELIST_END in new_dict:
            del new_dict[const.WHITELIST_END]
        return new_dict