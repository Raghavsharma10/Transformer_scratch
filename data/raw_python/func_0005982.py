def get_single_instance(sql, class_type, *args, **kwargs):
        """Returns an instance of class_type populated with attributes from the DB record; throws an error if no
        records are found

        @param sql: Sql statement to execute
        @param class_type: The type of class to instantiate and populate with DB record
        @return: Return an instance with attributes set to values from DB
        """
        record = CoyoteDb.get_single_record(sql, *args, **kwargs)
        try:
            instance = CoyoteDb.get_object_from_dictionary_representation(dictionary=record, class_type=class_type)
        except AttributeError:
            raise NoRecordsFoundException('No records found for {class_type} with sql run on {host}: \n {sql}'.format(
                sql=sql,
                host=DatabaseConfig().get('mysql_host'),
                class_type=class_type
            ))
        return instance