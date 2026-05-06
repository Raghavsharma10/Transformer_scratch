def insert_instance(instance, table, **kwargs):
        """Inserts an object's values into a given table, will not populate Nonetype values

        @param instance: Instance of an object to insert
        @param table: Table in which to insert instance values
        @return: ID of the last inserted row
        """
        instancedict = instance.__dict__.copy()
        instancedictclone = instancedict.copy()

        # Remove all Nonetype values
        for k, v in instancedictclone.iteritems():
            if v is None:
                instancedict.pop(k)

        keys, values = CoyoteDb.get_insert_fields_and_values_from_dict(instancedict)
        sql = """INSERT INTO {table} ({keys}) VALUES ({values})""".format(
            table=table,
            keys=keys,
            values=values
        )

        insert = CoyoteDb.insert(sql=sql, **kwargs)
        return insert