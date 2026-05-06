def record_to_objects(self):
        """Write from the stored file data to the source records"""
        from ambry.orm import SourceTable

        bsfile = self.record

        failures = set()

        # Clear out all of the columns from existing tables. We don't clear out the
        # tables, since they may be referenced by sources

        for row in bsfile.dict_row_reader:
            st = self._dataset.source_table(row['table'])

            if st:
                st.columns[:] = []

        self._dataset.commit()

        for row in bsfile.dict_row_reader:
            st = self._dataset.source_table(row['table'])

            if not st:
                st = self._dataset.new_source_table(row['table'])
                # table_number += 1

            if 'datatype' not in row:
                row['datatype'] = 'unknown'

            del row['table']

            st.add_column(**row)  # Create or update

        if failures:
            raise ConfigurationError('Failed to load source schema, missing sources: {} '.format(failures))

        self._dataset.commit()