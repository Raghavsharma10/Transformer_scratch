def get_functions_map(self):
        """Calculate the column name to data type conversion map"""
        return dict([(column, DATA_TYPE_FUNCTIONS[data_type]) for column, data_type in self.columns.values_list('name', 'data_type')])