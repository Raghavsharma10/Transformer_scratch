def _lookup_clublogAPI(self, callsign=None, timestamp=timestamp_now, url="https://secure.clublog.org/dxcc", apikey=None):
        """ Set up the Lookup object for Clublog Online API
        """

        params = {"year" : timestamp.strftime("%Y"),
            "month" : timestamp.strftime("%m"),
            "day" : timestamp.strftime("%d"),
            "hour" : timestamp.strftime("%H"),
            "minute" : timestamp.strftime("%M"),
            "api" : apikey,
            "full" : "1",
            "call" : callsign
        }

        if sys.version_info.major == 3:
            encodeurl = url + "?" + urllib.parse.urlencode(params)
        else:
            encodeurl = url + "?" + urllib.urlencode(params)
        response = requests.get(encodeurl, timeout=5)

        if not self._check_html_response(response):
            raise LookupError

        jsonLookup = response.json()
        lookup = {}

        for item in jsonLookup:
            if item == "Name": lookup[const.COUNTRY] = jsonLookup["Name"]
            elif item == "DXCC": lookup[const.ADIF] = int(jsonLookup["DXCC"])
            elif item == "Lon": lookup[const.LONGITUDE] = float(jsonLookup["Lon"])*(-1)
            elif item == "Lat": lookup[const.LATITUDE] = float(jsonLookup["Lat"])
            elif item == "CQZ": lookup[const.CQZ] = int(jsonLookup["CQZ"])
            elif item == "Continent": lookup[const.CONTINENT] = jsonLookup["Continent"]

        if lookup[const.ADIF] == 0:
            raise KeyError
        else:
            return lookup