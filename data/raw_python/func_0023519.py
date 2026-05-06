def create_get_property_request_content(option):
        """Creates an XML for requesting of getting a property value of remote WebDAV resource.

        :param option: the property attributes as dictionary with following keys:
                       `namespace`: (optional) the namespace for XML property which will be get,
                       `name`: the name of property which will be get.
        :return: the XML string of request content.
        """
        root = etree.Element('propfind', xmlns='DAV:')
        prop = etree.SubElement(root, 'prop')
        etree.SubElement(prop, option.get('name', ''), xmlns=option.get('namespace', ''))
        tree = etree.ElementTree(root)
        return WebDavXmlUtils.etree_to_string(tree)