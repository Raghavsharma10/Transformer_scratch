def _deserialize_data(self, json_data):
        """
        Deserialize a JSON into a dictionary
        """

        my_dict = json.loads(json_data.decode('utf8').replace("'", '"'),
            encoding='UTF-8')

        for item in my_dict:
            if item == const.ADIF:
                my_dict[item] = int(my_dict[item])
            elif item == const.DELETED:
                my_dict[item] = self._str_to_bool(my_dict[item])
            elif item == const.CQZ:
                my_dict[item] = int(my_dict[item])
            elif item == const.ITUZ:
                my_dict[item] = int(my_dict[item])
            elif item == const.LATITUDE:
                my_dict[item] = float(my_dict[item])
            elif item == const.LONGITUDE:
                my_dict[item] = float(my_dict[item])
            elif item == const.START:
                my_dict[item] = datetime.strptime(my_dict[item], '%Y-%m-%d%H:%M:%S').replace(tzinfo=UTC)
            elif item == const.END:
                my_dict[item] = datetime.strptime(my_dict[item], '%Y-%m-%d%H:%M:%S').replace(tzinfo=UTC)
            elif item == const.WHITELIST_START:
                my_dict[item] = datetime.strptime(my_dict[item], '%Y-%m-%d%H:%M:%S').replace(tzinfo=UTC)
            elif item == const.WHITELIST_END:
                my_dict[item] = datetime.strptime(my_dict[item], '%Y-%m-%d%H:%M:%S').replace(tzinfo=UTC)
            elif item == const.WHITELIST:
                my_dict[item] = self._str_to_bool(my_dict[item])
            else:
                my_dict[item] = unicode(my_dict[item])

        return my_dict