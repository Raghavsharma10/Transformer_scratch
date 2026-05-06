def _check_zone_exception_for_date(self, item, timestamp, data_dict, data_index_dict):
        """
        Checks the index and data if a cq-zone exception exists for the callsign
        When a zone exception is found, the zone is returned. If no exception is found
        a KeyError is raised

        """
        if item in data_index_dict:
            for item in data_index_dict[item]:

                # startdate < timestamp
                if const.START in data_dict[item] and not const.END in data_dict[item]:
                    if data_dict[item][const.START] < timestamp:
                        return data_dict[item][const.CQZ]

                # enddate > timestamp
                elif not const.START in data_dict[item] and const.END in data_dict[item]:
                    if data_dict[item][const.END] > timestamp:
                        return data_dict[item][const.CQZ]

                # startdate > timestamp > enddate
                elif const.START in data_dict[item] and const.END in data_dict[item]:
                    if data_dict[item][const.START] < timestamp \
                            and data_dict[item][const.END] > timestamp:
                        return data_dict[item][const.CQZ]

                # no startdate or enddate available
                elif not const.START in data_dict[item] and not const.END in data_dict[item]:
                        return data_dict[item][const.CQZ]

        raise KeyError