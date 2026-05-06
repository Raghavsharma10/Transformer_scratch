def info_parking_poi(self, **kwargs):
        """Obtain generic information on POIs and parkings.

        This returns a list of elements in a given radius from the coordinates.

        Args:
            radius (int): Radius of the search (in meters).
            latitude (double): Latitude in decimal degrees.
            longitude (double): Longitude in decimal degrees.
            lang (str): Language code (*es* or *en*).
            day (int): Day of the month in format DD.
                The number is automatically padded if it only has one digit.
            month (int): Month number in format MM.
                The number is automatically padded if it only has one digit.
            year (int): Year number in format YYYY.
            hour (int): Hour of the day in format hh.
                The number is automatically padded if it only has one digit.
            minute (int): Minute of the hour in format mm.
                The number is automatically padded if it only has one digit.
            poi_info (list[tuple]): List of tuples with the format
                ``(list[family], type, category)`` to query. Check the API
                documentation.
            min_free (list[int]): Number of free spaces to check. Must have the
                same length of ``poi_info``.
            field_codes (list[tuple]): List of tuples with the format
                ``(list[codes], name)``. Check the API documentation.

        Returns:
            Status boolean and parsed response (list[InfoParkingPoi]), or
            message string in case of error.
        """
        # Endpoint parameters
        date = util.datetime_string(
            kwargs.get('day', 1),
            kwargs.get('month', 1),
            kwargs.get('year', 1970),
            kwargs.get('hour', 0),
            kwargs.get('minute', 0)
        )

        family_categories = []
        for element in kwargs.get('poi_info', []):
            family_categories.append({
                'poiCategory': {
                    'lstCategoryTypes': element[0]
                    },
                'poiFamily': element[1],
                'poiType': element[2]
            })

        field_codes = []
        for element in kwargs.get('field_codes', []):
            field_codes.append({
                'codes': {
                    'lstCodes': element[0]
                    },
                'nameField': element[1]
            })

        params = {
            'TFamilyTTypeTCategory': {
                'lstFamilyTypeCategory': family_categories
            },
            'coordinate': {
                'latitude': str(kwargs.get('latitude', '0.0')),
                'longitude': str(kwargs.get('longitude', '0.0'))
            },
            'dateTimeUse': date,
            'language': util.language_code(kwargs.get('lang')),
            'minimumPlacesAvailable': {
                'lstminimumPlacesAvailable': kwargs.get('min_free', [])
            },
            'nameFieldCodes': {
                'lstNameFieldCodes': field_codes
            },
            'radius': str(kwargs.get('radius', '0'))
        }

        # Request
        result = self.make_request('info_parking_poi', {}, **params)

        if not util.check_result(result):
            return False, result.get('message', 'UNKNOWN ERROR')

        # Parse
        values = util.response_list(result, 'Data')
        return True, [emtype.InfoParkingPoi(**a) for a in values]