def normalize_operations(self, operations):
        """
        Removes redundant SQL operations - e.g. a CREATE X followed by a DROP X
        """
        normalized = OrderedDict()

        for operation in operations:
            op_key = (operation.sql_type, operation.obj_name)

            # do we already have an operation for this object?
            if op_key in normalized:
                if self.verbosity >= 2:
                    self.stdout.write(" < %s" % normalized[op_key])

                del normalized[op_key]

            # don't add DROP operations for objects not previously created
            if operation.is_create:
                normalized[op_key] = operation
            elif self.verbosity >= 2:
                self.stdout.write(" < %s" % operation)

        return normalized.values()