def load(self, *objs, consistent=False):
        """Populate objects from DynamoDB.

        :param objs: objects to delete.
        :param bool consistent: Use `strongly consistent reads`__ if True.  Default is False.
        :raises bloop.exceptions.MissingKey: if any object doesn't provide a value for a key column.
        :raises bloop.exceptions.MissingObjects: if one or more objects aren't loaded.

        __ http://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.ReadConsistency.html
        """
        get_table_name = self._compute_table_name
        objs = set(objs)
        validate_not_abstract(*objs)

        table_index, object_index, request = {}, {}, {}

        for obj in objs:
            table_name = get_table_name(obj.__class__)
            key = dump_key(self, obj)
            index = index_for(key)

            if table_name not in object_index:
                table_index[table_name] = list(sorted(key.keys()))
                object_index[table_name] = {}
                request[table_name] = {"Keys": [], "ConsistentRead": consistent}

            if index not in object_index[table_name]:
                request[table_name]["Keys"].append(key)
                object_index[table_name][index] = set()
            object_index[table_name][index].add(obj)

        response = self.session.load_items(request)

        for table_name, list_of_attrs in response.items():
            for attrs in list_of_attrs:
                key_shape = table_index[table_name]
                key = extract_key(key_shape, attrs)
                index = index_for(key)

                for obj in object_index[table_name].pop(index):
                    unpack_from_dynamodb(
                        attrs=attrs, expected=obj.Meta.columns, engine=self, obj=obj)
                    object_loaded.send(self, engine=self, obj=obj)
                if not object_index[table_name]:
                    object_index.pop(table_name)

        if object_index:
            not_loaded = set()
            for index in object_index.values():
                for index_set in index.values():
                    not_loaded.update(index_set)
            logger.info("loaded {} of {} objects".format(len(objs) - len(not_loaded), len(objs)))
            raise MissingObjects("Failed to load some objects.", objects=not_loaded)
        logger.info("successfully loaded {} objects".format(len(objs)))