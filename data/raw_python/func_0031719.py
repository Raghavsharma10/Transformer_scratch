def change_dict_keys(self, data_dict, prefix):
        """
            Prefixes 'L_'/'R_' to the collection keys
            :param data_dict: dictionary which is to be altered
            :type  data_dict: dict

            :param prefix: prefix to be attached before every key
            :type  prefix: string

            :return dict_: dict
        """

        keys = data_dict.keys()
        dummy_dict = copy.deepcopy(data_dict)
        changed_dict = {}
        for key in keys:
            changed_dict[prefix + str(key)] = dummy_dict.pop(key)
        return changed_dict