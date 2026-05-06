def create_table(self, table_name, model):
        """Create the model's table.  Returns True if the table is being created, False otherwise.

        Does not wait for the table to create, and does not validate an existing table.
        Will not raise "ResourceInUseException" if the table exists or is being created.

        :param str table_name: The name of the table to create for the model.
        :param model: The :class:`~bloop.models.BaseModel` to create the table for.
        :return: True if the table is being created, False if the table exists
        :rtype: bool
        """
        table = create_table_request(table_name, model)
        try:
            self.dynamodb_client.create_table(**table)
            is_creating = True
        except botocore.exceptions.ClientError as error:
            handle_table_exists(error, model)
            is_creating = False
        return is_creating