def _load_clublogXML(self,
                        url="https://secure.clublog.org/cty.php",
                        apikey=None,
                        cty_file=None):
        """ Load and process the ClublogXML file either as a download or from file
        """

        if self._download:
            cty_file = self._download_file(
                    url = url,
                    apikey = apikey)
        else:
            cty_file = self._lib_filename

        header = self._extract_clublog_header(cty_file)
        cty_file = self._remove_clublog_xml_header(cty_file)
        cty_dict = self._parse_clublog_xml(cty_file)

        self._entities = cty_dict["entities"]
        self._callsign_exceptions = cty_dict["call_exceptions"]
        self._prefixes = cty_dict["prefixes"]
        self._invalid_operations = cty_dict["invalid_operations"]
        self._zone_exceptions = cty_dict["zone_exceptions"]

        self._callsign_exceptions_index = cty_dict["call_exceptions_index"]
        self._prefixes_index = cty_dict["prefixes_index"]
        self._invalid_operations_index = cty_dict["invalid_operations_index"]
        self._zone_exceptions_index = cty_dict["zone_exceptions_index"]

        return True