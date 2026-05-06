def add_class_properties_xsd(self, tb, cls):
        """ Add the XSD for the class properties to the ``TreeBuilder``. And
        call the user ``sequence_callback``. """
        for p in class_mapper(cls).iterate_properties:
            if isinstance(p, ColumnProperty):
                self.add_column_property_xsd(tb, p)
        if self.sequence_callback:
            self.sequence_callback(tb, cls)