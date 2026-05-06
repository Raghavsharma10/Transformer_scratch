def _xml_pretty_print(self, data):
        """Pretty print xml data
        """
        raw_string = xtree.tostring(data, 'utf-8')
        parsed_string = minidom.parseString(raw_string)
        return parsed_string.toprettyxml(indent='\t')