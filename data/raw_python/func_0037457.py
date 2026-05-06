def import_table(self, table):
        """import table by Table object
        
        :param `manager.table_conf.Table` table: Table object
        """
        file_path = os.path.join(self.pyctd_data_dir, table.file_name)
        log.info('importing %s data into table %s', file_path, table.name)
        table_import_timer = time.time()

        use_columns_with_index, column_names_in_db = self.get_index_and_columns_order(
            table.columns_in_file_expected,
            table.columns_dict,
            file_path
        )

        self.import_table_in_db(file_path, use_columns_with_index, column_names_in_db, table)

        for column_in_file, column_in_one2many_table in table.one_to_many:
            o2m_column_index = self.get_index_of_column(column_in_file, file_path)

            self.import_one_to_many(file_path, o2m_column_index, table, column_in_one2many_table)

        log.info('done importing %s in %.2f seconds', table.name, time.time() - table_import_timer)