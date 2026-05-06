def import_one_to_many(self, file_path, column_index, parent_table, column_in_one2many_table):
        """
        
        :param file_path: 
        :param column_index: 
        :param parent_table:
        :param column_in_one2many_table: 
        """
        chunks = pd.read_table(
            file_path,
            usecols=[column_index],
            header=None,
            comment='#',
            index_col=False,
            chunksize=1000000,
            dtype=self.get_dtypes(parent_table.model)
        )

        for chunk in chunks:
            child_values = []
            parent_id_values = []

            chunk.dropna(inplace=True)
            chunk.index += 1

            for parent_id, values in chunk.iterrows():
                entry = values[column_index]
                if not isinstance(entry, str):
                    entry = str(entry)
                for value in entry.split("|"):
                    parent_id_values.append(parent_id)
                    child_values.append(value.strip())

            parent_id_column_name = parent_table.name + '__id'
            o2m_table_name = defaults.TABLE_PREFIX + parent_table.name + '__' + column_in_one2many_table

            pd.DataFrame({
                parent_id_column_name: parent_id_values,
                column_in_one2many_table: child_values
            }).to_sql(name=o2m_table_name, if_exists='append', con=self.engine, index=False)