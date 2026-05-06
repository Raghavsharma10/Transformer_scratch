def _extract_clublog_header(self, cty_xml_filename):
        """
        Extract the header of the Clublog XML File
        """

        cty_header = {}

        try:
            with open(cty_xml_filename, "r") as cty:
                raw_header = cty.readline()

            cty_date = re.search("date='.+'", raw_header)
            if cty_date:
                cty_date = cty_date.group(0).replace("date=", "").replace("'", "")
                cty_date = datetime.strptime(cty_date[:19], '%Y-%m-%dT%H:%M:%S')
                cty_date.replace(tzinfo=UTC)
                cty_header["Date"] = cty_date

            cty_ns = re.search("xmlns='.+[']", raw_header)
            if cty_ns:
                cty_ns = cty_ns.group(0).replace("xmlns=", "").replace("'", "")
                cty_header['NameSpace'] = cty_ns

            if len(cty_header) == 2:
                self._logger.debug("Header successfully retrieved from CTY File")
            elif len(cty_header) < 2:
                self._logger.warning("Header could only be partically retrieved from CTY File")
                self._logger.warning("Content of Header: ")
                for key in cty_header:
                    self._logger.warning(str(key)+": "+str(cty_header[key]))
            return cty_header

        except Exception as e:
            self._logger.error("Clublog CTY File could not be opened / modified")
            self._logger.error("Error Message: " + str(e))
            return