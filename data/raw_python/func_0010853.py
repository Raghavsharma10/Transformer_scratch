def write_type_dumps(self, operations, preserve_order, output_dir):
        """
        Splits the list of SQL operations by type and dumps these to separate files
        """
        by_type = {SqlType.INDEX: [], SqlType.FUNCTION: [], SqlType.TRIGGER: []}
        for operation in operations:
            by_type[operation.sql_type].append(operation)

        # optionally sort each operation list by the object name
        if not preserve_order:
            for obj_type, ops in by_type.items():
                by_type[obj_type] = sorted(ops, key=lambda o: o.obj_name)

        if by_type[SqlType.INDEX]:
            self.write_dump('indexes', by_type[SqlType.INDEX], output_dir)
        if by_type[SqlType.FUNCTION]:
            self.write_dump('functions', by_type[SqlType.FUNCTION], output_dir)
        if by_type[SqlType.TRIGGER]:
            self.write_dump('triggers', by_type[SqlType.TRIGGER], output_dir)