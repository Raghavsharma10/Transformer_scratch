def _load_countryfile(self,
                         url="https://www.country-files.com/cty/cty.plist",
                         country_mapping_filename="countryfilemapping.json",
                         cty_file=None):
        """ Load and process the ClublogXML file either as a download or from file
        """

        cwdFile = os.path.abspath(os.path.join(os.getcwd(), country_mapping_filename))
        pkgFile = os.path.abspath(os.path.join(os.path.dirname(__file__), country_mapping_filename))

        # from cwd
        if os.path.exists(cwdFile):
            # country mapping files contains the ADIF identifiers of a particular
            # country since the country-files do not provide this information (only DXCC id)
            country_mapping_filename = cwdFile
        # from package
        elif os.path.exists(pkgFile):
            country_mapping_filename = pkgFile
        else:
            country_mapping_filename = None

        if self._download:
            cty_file = self._download_file(url=url)
        else:
            cty_file = os.path.abspath(cty_file)

        cty_dict = self._parse_country_file(cty_file, country_mapping_filename)
        self._callsign_exceptions = cty_dict["exceptions"]
        self._prefixes = cty_dict["prefixes"]
        self._callsign_exceptions_index = cty_dict["exceptions_index"]
        self._prefixes_index = cty_dict["prefixes_index"]

        return True