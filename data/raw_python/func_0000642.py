def restore_bucket(self, table_name):
        """Restore bucket from SQL
        """
        if table_name.startswith(self.__prefix):
            return table_name.replace(self.__prefix, '', 1)
        return None