def import_table_in_db(self, file_path, use_columns_with_index, column_names_in_db, table):
        """Imports data from CTD file into database
        
        :param str file_path: path to file
        :param list[int] use_columns_with_index: list of column indices in file
        :param list[str] column_names_in_db: list of column names (have to fit to models except domain_id column name)
        :param table: `manager.table.Table` object
        """
        chunks = pd.read_table(
            file_path,
            usecols=use_columns_with_index,
            names=column_names_in_db,
            header=None, comment='#',
            index_col=False,
            chunksize=1000000,
            dtype=self.get_dtypes(table.model)
        )

        for chunk in chunks:
            # this is an evil hack because CTD is not using the MESH prefix in this table
            if table.name == 'exposure_event':
                chunk.disease_id = 'MESH:' + chunk.disease_id

            chunk['id'] = chunk.index + 1

            if table.model not in table_conf.models_to_map:
                for model in table_conf.models_to_map:
                    domain = model.table_suffix
                    domain_id = domain + "_id"
                    if domain_id in column_names_in_db:
                        chunk = pd.merge(chunk, self.mapper[domain], on=domain_id, how='left')
                        del chunk[domain_id]

            chunk.set_index('id', inplace=True)

            table_with_prefix = defaults.TABLE_PREFIX + table.name
            chunk.to_sql(name=table_with_prefix, if_exists='append', con=self.engine)

        del chunks