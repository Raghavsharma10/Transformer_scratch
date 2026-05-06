def create_set_property_batch_request_content(options):
        """Creates an XML for requesting of setting a property values for remote WebDAV resource in batch.

        :param options: the property attributes as list of dictionaries with following keys:
                       `namespace`: (optional) the namespace for XML property which will be set,
                       `name`: the name of property which will be set,
                       `value`: (optional) the value of property which will be set. Defaults is empty string.
        :return: the XML string of request content.
        """
        root_node = etree.Element('propertyupdate', xmlns='DAV:')
        set_node = etree.SubElement(root_node, 'set')
        prop_node = etree.SubElement(set_node, 'prop')
        for option in options:
            opt_node = etree.SubElement(prop_node, option['name'], xmlns=option.get('namespace', ''))
            opt_node.text = option.get('value', '')
        tree = etree.ElementTree(root_node)
        return WebDavXmlUtils.etree_to_string(tree)