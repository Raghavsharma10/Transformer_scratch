def handle_typed_values(val, type_name, value_type):
        """Translate typed values into the appropriate python object.

        Takes an element name, value, and type and returns a list
        with the string value(s) properly converted to a python type.

        TypedValues are handled in ucar.ma2.DataType in netcdfJava
        in the DataType enum. Possibilities are:

            "boolean"
            "byte"
            "char"
            "short"
            "int"
            "long"
            "float"
            "double"
            "Sequence"
            "String"
            "Structure"
            "enum1"
            "enum2"
            "enum4"
            "opaque"
            "object"

        All of these are values written as strings in the xml, so simply
        applying int, float to the values will work in most cases (i.e.
        the TDS encodes them as string values properly).

        Examle XML element:

        <attribute name="scale_factor" type="double" value="0.0010000000474974513"/>

        Parameters
        ----------
        val : string
            The string representation of the value attribute of the xml element

        type_name : string
            The string representation of the name attribute of the xml element

        value_type : string
            The string representation of the type attribute of the xml element

        Returns
        -------
        val : list
            A list containing the properly typed python values.

        """
        if value_type in ['byte', 'short', 'int', 'long']:
            try:
                val = [int(v) for v in re.split('[ ,]', val) if v]
            except ValueError:
                log.warning('Cannot convert "%s" to int. Keeping type as str.', val)
        elif value_type in ['float', 'double']:
            try:
                val = [float(v) for v in re.split('[ ,]', val) if v]
            except ValueError:
                log.warning('Cannot convert "%s" to float. Keeping type as str.', val)
        elif value_type == 'boolean':
            try:
                # special case for boolean type
                val = val.split()
                # values must be either true or false
                for potential_bool in val:
                    if potential_bool not in ['true', 'false']:
                        raise ValueError
                val = [True if item == 'true' else False for item in val]
            except ValueError:
                msg = 'Cannot convert values %s to boolean.'
                msg += ' Keeping type as str.'
                log.warning(msg, val)
        elif value_type == 'String':
            # nothing special for String type
            pass
        else:
            # possibilities - Sequence, Structure, enum, opaque, object,
            # and char.
            # Not sure how to handle these as I do not have an example
            # of how they would show up in dataset.xml
            log.warning('%s type %s not understood. Keeping as String.',
                        type_name, value_type)

        if not isinstance(val, list):
            val = [val]

        return val