def __get_table(self, bucket):
        """Get table by bucket
        """
        table_name = self.__mapper.convert_bucket(bucket)
        if self.__dbschema:
            table_name = '.'.join((self.__dbschema, table_name))
        return self.__metadata.tables[table_name]