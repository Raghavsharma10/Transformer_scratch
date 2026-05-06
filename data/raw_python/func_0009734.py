def __init_xml(self, rootElementTag):
        """Init a etree element and pop a key in there"""
        xml_root = etree.Element(rootElementTag)
        key = etree.SubElement(xml_root, "Key")
        key.text = self.apikey
        return xml_root