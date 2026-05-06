def _check_data_for_date(self, item, timestamp, data_dict, data_index_dict):
        """
        Checks if the item is found in the index. An entry in the index points to the data
        in the data_dict. This is mainly used retrieve callsigns and prefixes.
        In case data is found for item, a dict containing the data is returned. Otherwise a KeyError is raised.
        """

        if item in data_index_dict:
            for item in data_index_dict[item]:

                # startdate < timestamp
                if const.START in data_dict[item] and not const.END in data_dict[item]:
                    if data_dict[item][const.START] < timestamp:
                        item_data = copy.deepcopy(data_dict[item])
                        del item_data[const.START]
                        return item_data

                # enddate > timestamp
                elif not const.START in data_dict[item] and const.END in data_dict[item]:
                    if data_dict[item][const.END] > timestamp:
                        item_data = copy.deepcopy(data_dict[item])
                        del item_data[const.END]
                        return item_data

                # startdate > timestamp > enddate
                elif const.START in data_dict[item] and const.END in data_dict[item]:
                    if data_dict[item][const.START] < timestamp \
                            and data_dict[item][const.END] > timestamp:
                        item_data = copy.deepcopy(data_dict[item])
                        del item_data[const.START]
                        del item_data[const.END]
                        return item_data

                # no startdate or enddate available
                elif not const.START in data_dict[item] and not const.END in data_dict[item]:
                    return data_dict[item]

        raise KeyError