def _handle_hdr(self, hdr):
        """Given the file header line (or one provided when the object
        is instantiated) this method populates the ``self._converters`` array,
        a list of type converters indexed by the column name.

        :param hdr: The header line.

        :raises: ContentError for any formatting problems
        :raises: UnknownTypeError if the type is not known
        """

        column_number = 1
        for cell in hdr:
            cell_parts = cell.split(self._type_sep)
            if len(cell_parts) not in [1, 2]:
                raise ContentError(column_number, self._c_reader.line_num,
                                   cell, 'Expected name and type (up to 2 items)')
            name = cell_parts[0].strip()
            if len(name) == 0:
                raise ContentError(column_number, self._c_reader.line_num,
                                   cell, 'Column name is empty')
            if name in self._column_names:
                raise ContentError(column_number, self._c_reader.line_num,
                                   name, 'Duplicate column name')

            if len(cell_parts) == 2:
                column_type = cell_parts[1].strip().lower()
                if column_type not in CONVERTERS:
                    raise UnknownTypeError(column_number, column_type)
            else:
                # Unspecified - assume built-in 'string'
                column_type = 'string'
            self._converters.append([name, CONVERTERS[column_type]])
            self._column_names.append(name)
            column_number += 1