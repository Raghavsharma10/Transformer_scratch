def add_column_xsd(self, tb, column, attrs):
        """ Add the XSD for a column to tb (a TreeBuilder) """
        if column.nullable:
            attrs['minOccurs'] = str(0)
            attrs['nillable'] = 'true'
        for cls, xsd_type in six.iteritems(self.SIMPLE_XSD_TYPES):
            if isinstance(column.type, cls):
                attrs['type'] = xsd_type
                with tag(tb, 'xsd:element', attrs) as tb:
                    self.element_callback(tb, column)
                    return tb
        if isinstance(column.type, Geometry):
            geometry_type = column.type.geometry_type
            xsd_type = self.SIMPLE_GEOMETRY_XSD_TYPES[geometry_type]
            attrs['type'] = xsd_type
            with tag(tb, 'xsd:element', attrs) as tb:
                self.element_callback(tb, column)
                return tb
        if isinstance(column.type, sqlalchemy.Enum):
            with tag(tb, 'xsd:element', attrs) as tb:
                with tag(tb, 'xsd:simpleType') as tb:
                    with tag(tb, 'xsd:restriction', {'base': 'xsd:string'}) \
                            as tb:
                        for enum in column.type.enums:
                            with tag(tb, 'xsd:enumeration', {'value': enum}):
                                pass
                self.element_callback(tb, column)
                return tb
        if isinstance(column.type, sqlalchemy.Numeric):
            if column.type.scale is None and column.type.precision is None:
                attrs['type'] = 'xsd:decimal'
                with tag(tb, 'xsd:element', attrs) as tb:
                    self.element_callback(tb, column)
                    return tb
            else:
                with tag(tb, 'xsd:element', attrs) as tb:
                    with tag(tb, 'xsd:simpleType') as tb:
                        with tag(tb, 'xsd:restriction',
                                 {'base': 'xsd:decimal'}) as tb:
                            if column.type.scale is not None:
                                with tag(tb, 'xsd:fractionDigits',
                                         {'value': str(column.type.scale)}) \
                                        as tb:
                                    pass
                            if column.type.precision is not None:
                                precision = column.type.precision
                                with tag(tb, 'xsd:totalDigits',
                                         {'value': str(precision)}) \
                                        as tb:
                                    pass
                    self.element_callback(tb, column)
                    return tb
        if isinstance(column.type, sqlalchemy.String) \
                or isinstance(column.type, sqlalchemy.Text) \
                or isinstance(column.type, sqlalchemy.Unicode) \
                or isinstance(column.type, sqlalchemy.UnicodeText):
            if column.type.length is None:
                attrs['type'] = 'xsd:string'
                with tag(tb, 'xsd:element', attrs) as tb:
                    self.element_callback(tb, column)
                    return tb
            else:
                with tag(tb, 'xsd:element', attrs) as tb:
                    with tag(tb, 'xsd:simpleType') as tb:
                        with tag(tb, 'xsd:restriction',
                                 {'base': 'xsd:string'}) as tb:
                            with tag(tb, 'xsd:maxLength',
                                     {'value': str(column.type.length)}):
                                pass
                    self.element_callback(tb, column)
                    return tb
        raise UnsupportedColumnTypeError(column.type)