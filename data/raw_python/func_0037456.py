def get_index_and_columns_order(cls, columns_in_file_expected, columns_dict, file_path):
        """
        
        :param columns_in_file_expected: 
        :param columns_dict: 
        :param file_path: 
        :rtype: tuple[list,list]
        """
        use_columns_with_index = []
        column_names_in_db = []

        column_names_from_file = cls.get_column_names_from_file(file_path)
        if not set(columns_in_file_expected).issubset(column_names_from_file):
            log.exception(
                '%s columns are not a subset of columns %s in file %s',
                columns_in_file_expected,
                column_names_from_file,
                file_path
            )
        else:
            for index, column in enumerate(column_names_from_file):
                if column in columns_dict:
                    use_columns_with_index.append(index)
                    column_names_in_db.append(columns_dict[column])
        return use_columns_with_index, column_names_in_db