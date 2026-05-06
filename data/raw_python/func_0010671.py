def _get_dicts_from_redis(self, name, index_name, redis_prefix, item):
        """
        Retrieve the data of an item from redis and put it in an index and data dictionary to match the
        common query interface.
        """
        r = self._redis
        data_dict = {}
        data_index_dict = {}

        if redis_prefix is None:
            raise KeyError ("redis_prefix is missing")

        if r.scard(redis_prefix + index_name + str(item)) > 0:
            data_index_dict[str(item)] = r.smembers(redis_prefix + index_name + str(item))

            for i in data_index_dict[item]:
                json_data = r.get(redis_prefix + name + str(int(i)))
                data_dict[i] = self._deserialize_data(json_data)

            return (data_dict, data_index_dict)

        raise KeyError ("No Data found in Redis for "+ item)