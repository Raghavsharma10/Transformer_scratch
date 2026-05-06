def lookup_entity(self, entity=None):
        """Returns lookup data of an ADIF Entity

        Args:
            entity (int): ADIF identifier of country

        Returns:
            dict: Dictionary containing the country specific data

        Raises:
            KeyError: No matching entity found

        Example:
           The following code queries the the Clublog XML database for the ADIF entity Turkmenistan, which has
           the id 273.

           >>> from pyhamtools import LookupLib
           >>> my_lookuplib = LookupLib(lookuptype="clublogapi", apikey="myapikey")
           >>> print my_lookuplib.lookup_entity(273)
           {
            'deleted': False,
            'country': u'TURKMENISTAN',
            'longitude': 58.4,
            'cqz': 17,
            'prefix': u'EZ',
            'latitude': 38.0,
            'continent': u'AS'
           }


        Note:
            This method is available for the following lookup type

            - clublogxml
            - redis
            - qrz.com

        """
        if self._lookuptype == "clublogxml":
            entity = int(entity)
            if entity in self._entities:
                return self._strip_metadata(self._entities[entity])
            else:
                raise KeyError

        elif self._lookuptype == "redis":
            if self._redis_prefix is None:
                raise KeyError ("redis_prefix is missing")
            #entity = str(entity)
            json_data = self._redis.get(self._redis_prefix + "_entity_" + str(entity))
            if json_data is not None:
                my_dict = self._deserialize_data(json_data)
                return self._strip_metadata(my_dict)

        elif self._lookuptype == "qrz":
            result = self._lookup_qrz_dxcc(entity, self._apikey)
            return result

        # no matching case
        raise KeyError