def get_class_xsd(self, io, cls):
        """ Returns the XSD for a mapped class. """
        attrs = {}
        attrs['xmlns:gml'] = 'http://www.opengis.net/gml'
        attrs['xmlns:xsd'] = 'http://www.w3.org/2001/XMLSchema'
        tb = TreeBuilder()
        with tag(tb, 'xsd:schema', attrs) as tb:
            with tag(tb, 'xsd:complexType', {'name': cls.__name__}) as tb:
                with tag(tb, 'xsd:complexContent') as tb:
                    with tag(tb, 'xsd:extension',
                             {'base': 'gml:AbstractFeatureType'}) as tb:
                        with tag(tb, 'xsd:sequence') as tb:
                            self.add_class_properties_xsd(tb, cls)

        ElementTree(tb.close()).write(io, encoding='utf-8')
        return io