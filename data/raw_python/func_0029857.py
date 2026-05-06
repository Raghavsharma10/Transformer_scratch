def record_to_objects(self):
        """Create config records to match the file metadata"""
        from ambry.orm import Column, Table, Dataset

        def _clean_int(i):
            if i is None:
                return None
            elif isinstance(i, int):
                return i
            elif isinstance(i, string_types):
                if len(i) == 0:
                    return None

                return int(i.strip())

        bsfile = self.record

        contents = bsfile.unpacked_contents

        if not contents:
            return

        line_no = 1  # Accounts for file header. Data starts on line 2

        errors = []
        warnings = []

        extant_tables = {t.name: t for t in self._dataset.tables}

        old_types_map = {
            'varchar': Column.DATATYPE_STR,
            'integer': Column.DATATYPE_INTEGER,
            'real': Column.DATATYPE_FLOAT,
        }

        def run_progress_f(line_no):
            self._bundle.log('Loading tables from file. Line #{}'.format(line_no))

        from ambry.bundle.process import CallInterval
        run_progress_f = CallInterval(run_progress_f, 10)

        table_number = self._dataset._database.next_sequence_id(Dataset, self._dataset.vid, Table)
        for row in bsfile.dict_row_reader:

            line_no += 1

            run_progress_f(line_no)

            # Skip blank lines
            if not row.get('column', False) and not row.get('table', False):
                continue

            if not row.get('column', False):
                raise ConfigurationError('Row error: no column on line {}'.format(line_no))

            if not row.get('table', False):
                raise ConfigurationError('Row error: no table on line {}'.format(line_no))

            if not row.get('datatype', False) and not row.get('valuetype', False):
                raise ConfigurationError('Row error: no type on line {}'.format(line_no))

            value_type = row.get('valuetype', '').strip() if row.get('valuetype', False) else None
            data_type = row.get('datatype', '').strip() if row.get('datatype', False) else None

            def resolve_data_type(value_type):
                from ambry.valuetype import resolve_value_type
                vt_class = resolve_value_type(value_type)

                if not vt_class:
                    raise ConfigurationError("Row error: unknown valuetype '{}'".format(value_type))

                return vt_class.python_type().__name__

            # If we have a value type field, and not the datatype,
            # the value type is as specified, and the data type is derived from it.
            if value_type and not data_type:
                data_type = resolve_data_type(value_type)

            elif data_type and not value_type:
                value_type = data_type
                data_type = resolve_data_type(value_type)

            # There are still some old data types hanging around
            data_type = old_types_map.get(data_type.lower(), data_type)

            table_name = row['table']

            try:
                table = extant_tables[table_name]
            except KeyError:
                table = self._dataset.new_table(
                    table_name,
                    sequence_id=table_number,
                    description=row.get('description') if row['column'] == 'id' else ''
                )

                table_number += 1
                extant_tables[table_name] = table

            data = {k.replace('d_', '', 1): v
                    for k, v in list(row.items()) if k and k.startswith('d_') and v}

            if row['column'] == 'id':
                table.data.update(data)
                data = {}

            table.add_column(
                row['column'],
                fk_vid=row['is_fk'] if row.get('is_fk', False) else None,
                description=(row.get('description', '') or '').strip(),
                datatype=data_type,
                valuetype=value_type,
                parent=row.get('parent'),
                proto_vid=row.get('proto_vid'),
                size=_clean_int(row.get('size', None)),
                width=_clean_int(row.get('width', None)),
                data=data,
                keywords=row.get('keywords'),
                measure=row.get('measure'),
                transform=row.get('transform'),
                derivedfrom=row.get('derivedfrom'),
                units=row.get('units', None),
                universe=row.get('universe'),
                update_existing= True)

        self._dataset.t_sequence_id = table_number

        return warnings, errors