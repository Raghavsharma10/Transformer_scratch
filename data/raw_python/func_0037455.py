def get_index_of_column(cls, column, file_path):
        """Get index of a specific column name in a CTD file
        
        :param column: 
        :param file_path: 
        :return: Optional[int]
        """
        columns = cls.get_column_names_from_file(file_path)
        if column in columns:
            return columns.index(column)