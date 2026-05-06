def open_ioc(fn):
        """
        Opens an IOC file, or XML string.  Returns the root element, top level
        indicator element, and parameters element.  If the IOC or string fails
        to parse, an IOCParseError is raised.

        This is a helper function used by __init__.

        :param fn: This is a path to a file to open, or a string containing XML representing an IOC.
        :return: a tuple containing three elementTree Element objects
         The first element, the root, contains the entire IOC itself.
         The second element, the top level OR indicator, allows the user to add
          additional IndicatorItem or Indicator nodes to the IOC easily.
         The third element, the parameters node, allows the user to quickly
          parse the parameters.
        """
        parsed_xml = xmlutils.read_xml_no_ns(fn)
        if not parsed_xml:
            raise IOCParseError('Error occured parsing XML')
        root = parsed_xml.getroot()
        metadata_node = root.find('metadata')
        top_level_indicator = get_top_level_indicator_node(root)
        parameters_node = root.find('parameters')
        if parameters_node is None:
            # parameters node is not required by schema; but we add it if it is not present
            parameters_node = ioc_et.make_parameters_node()
            root.append(parameters_node)
        return root, metadata_node, top_level_indicator, parameters_node