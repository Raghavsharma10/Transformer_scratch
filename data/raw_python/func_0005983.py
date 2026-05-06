def get_all_instances(sql, class_type, *args, **kwargs):
        """Returns a list of instances of class_type populated with attributes from the DB record

        @param sql: Sql statement to execute
        @param class_type: The type of class to instantiate and populate with DB record
        @return: Return a list of instances with attributes set to values from DB
        """
        records = CoyoteDb.get_all_records(sql, *args, **kwargs)
        instances = [CoyoteDb.get_object_from_dictionary_representation(
            dictionary=record, class_type=class_type) for record in records]
        for instance in instances:
            instance._query = sql
        return instances