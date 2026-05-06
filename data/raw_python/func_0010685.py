def _remove_clublog_xml_header(self, cty_xml_filename):
        """
            remove the header of the Clublog XML File to make it
            properly parseable for the python ElementTree XML parser
        """
        import tempfile

        try:
            with open(cty_xml_filename, "r") as f:
                content = f.readlines()

            cty_dir = tempfile.gettempdir()
            cty_name = os.path.split(cty_xml_filename)[1]
            cty_xml_filename_no_header = os.path.join(cty_dir, "NoHeader_"+cty_name)

            with open(cty_xml_filename_no_header, "w") as f:
                f.writelines("<clublog>\n\r")
                f.writelines(content[1:])

            self._logger.debug("Header successfully modified for XML Parsing")
            return cty_xml_filename_no_header

        except Exception as e:
            self._logger.error("Clublog CTY could not be opened / modified")
            self._logger.error("Error Message: " + str(e))
            return