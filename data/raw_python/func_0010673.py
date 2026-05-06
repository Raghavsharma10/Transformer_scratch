def _check_inv_operation_for_date(self, item, timestamp, data_dict, data_index_dict):
        """
        Checks if the callsign is marked as an invalid operation for a given timestamp.
        In case the operation is invalid, True is returned. Otherwise a KeyError is raised.
        """

        if item in data_index_dict:
            for item in data_index_dict[item]:

                # startdate < timestamp
                if const.START in data_dict[item] and not const.END in data_dict[item]:
                    if data_dict[item][const.START] < timestamp:
                        return True

                # enddate > timestamp
                elif not const.START in data_dict[item] and const.END in data_dict[item]:
                    if data_dict[item][const.END] > timestamp:
                        return True

                # startdate > timestamp > enddate
                elif const.START in data_dict[item] and const.END in data_dict[item]:
                    if data_dict[item][const.START] < timestamp \
                            and data_dict[item][const.END] > timestamp:
                        return True

                # no startdate or enddate available
                elif not const.START in data_dict[item] and not const.END in data_dict[item]:
                    return True

        raise KeyError