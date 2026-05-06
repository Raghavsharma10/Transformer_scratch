def copy_data_in_redis(self, redis_prefix, redis_instance):
        """
        Copy the complete lookup data into redis. Old data will be overwritten.

        Args:
            redis_prefix (str): Prefix to distinguish the data in redis for the different looktypes
            redis_instance (str): an Instance of Redis

        Returns:
            bool: returns True when the data has been copied successfully into Redis

        Example:
           Copy the entire lookup data from the Country-files.com PLIST File into Redis. This example requires a running
           instance of Redis, as well the python Redis connector (pip install redis-py).

           >>> from pyhamtools import LookupLib
           >>> import redis
           >>> r = redis.Redis()
           >>> my_lookuplib = LookupLib(lookuptype="countryfile")
           >>> print my_lookuplib.copy_data_in_redis(redis_prefix="CF", redis_instance=r)
           True

           Now let's create an instance of LookupLib, using Redis to query the data

           >>> from pyhamtools import LookupLib
           >>> import redis
           >>> r = redis.Redis()
           >>> my_lookuplib = LookupLib(lookuptype="countryfile", redis_instance=r, redis_prefix="CF")
           >>> my_lookuplib.lookup_callsign("3D2RI")
           {
             u'adif': 460,
             u'continent': u'OC',
             u'country': u'Rotuma Island',
             u'cqz': 32,
             u'ituz': 56,
             u'latitude': -12.48,
             u'longitude': 177.08
           }


        Note:
            This method is available for the following lookup type

            - clublogxml
            - countryfile
        """

        if redis_instance is not None:
            self._redis = redis_instance

        if self._redis is None:
            raise AttributeError("redis_instance is missing")

        if redis_prefix is None:
            raise KeyError("redis_prefix is missing")

        if self._lookuptype == "clublogxml" or self._lookuptype == "countryfile":

            self._push_dict_to_redis(self._entities, redis_prefix, "_entity_")

            self._push_dict_index_to_redis(self._callsign_exceptions_index, redis_prefix, "_call_ex_index_")
            self._push_dict_to_redis(self._callsign_exceptions, redis_prefix, "_call_ex_")

            self._push_dict_index_to_redis(self._prefixes_index, redis_prefix, "_prefix_index_")
            self._push_dict_to_redis(self._prefixes, redis_prefix, "_prefix_")

            self._push_dict_index_to_redis(self._invalid_operations_index, redis_prefix, "_inv_op_index_")
            self._push_dict_to_redis(self._invalid_operations, redis_prefix, "_inv_op_")

            self._push_dict_index_to_redis(self._zone_exceptions_index, redis_prefix, "_zone_ex_index_")
            self._push_dict_to_redis(self._zone_exceptions, redis_prefix, "_zone_ex_")

        return True