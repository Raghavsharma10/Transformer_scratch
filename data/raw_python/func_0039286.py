def _create_table_xml_file(self, data, fname=None):
        """Creates a xml file of the table
        """
        content = self._xml_pretty_print(data)
        if not fname:
            fname = self.name
        with open(fname+".xml", 'w') as f:
            f.write(content)