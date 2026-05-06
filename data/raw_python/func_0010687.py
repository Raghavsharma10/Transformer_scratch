def _parse_country_file(self, cty_file, country_mapping_filename=None):
        """
        Parse the content of a PLIST file from country-files.com return the
        parsed values in dictionaries.
        Country-files.com provides Prefixes and Exceptions

        """

        import plistlib

        cty_list = None
        entities = {}
        exceptions = {}
        prefixes = {}

        exceptions_index = {}
        prefixes_index = {}

        exceptions_counter = 0
        prefixes_counter = 0

        mapping = None

        with open(country_mapping_filename, "r") as f:
            mapping = json.loads(f.read(),encoding='UTF-8')

        cty_list = plistlib.readPlist(cty_file)

        for item in cty_list:
            entry = {}
            call = str(item)
            entry[const.COUNTRY] = unicode(cty_list[item]["Country"])
            if mapping:
                 entry[const.ADIF] = int(mapping[cty_list[item]["Country"]])
            entry[const.CQZ] = int(cty_list[item]["CQZone"])
            entry[const.ITUZ] = int(cty_list[item]["ITUZone"])
            entry[const.CONTINENT] = unicode(cty_list[item]["Continent"])
            entry[const.LATITUDE] = float(cty_list[item]["Latitude"])
            entry[const.LONGITUDE] = float(cty_list[item]["Longitude"])*(-1)

            if cty_list[item]["ExactCallsign"]:
                if call in exceptions_index.keys():
                    exceptions_index[call].append(exceptions_counter)
                else:
                    exceptions_index[call] = [exceptions_counter]
                exceptions[exceptions_counter] = entry
                exceptions_counter += 1
            else:
                if call in prefixes_index.keys():
                    prefixes_index[call].append(prefixes_counter)
                else:
                    prefixes_index[call] = [prefixes_counter]
                prefixes[prefixes_counter] = entry
                prefixes_counter += 1

        self._logger.debug(str(len(prefixes))+" Prefixes added")
        self._logger.debug(str(len(prefixes_index))+" Prefixes in Index")
        self._logger.debug(str(len(exceptions))+" Exceptions added")
        self._logger.debug(str(len(exceptions_index))+" Exceptions in Index")

        result = {
            "prefixes" : prefixes,
            "exceptions" : exceptions,
            "prefixes_index" : prefixes_index,
            "exceptions_index" : exceptions_index,
        }

        return result