def create_free_space_request_content():
        """Creates an XML for requesting of free space on remote WebDAV server.

        :return: the XML string of request content.
        """
        root = etree.Element('propfind', xmlns='DAV:')
        prop = etree.SubElement(root, 'prop')
        etree.SubElement(prop, 'quota-available-bytes')
        etree.SubElement(prop, 'quota-used-bytes')
        tree = etree.ElementTree(root)
        return WebDavXmlUtils.etree_to_string(tree)