def restore_descriptor(self, table_name, columns, constraints, autoincrement_column=None):
        """Restore descriptor from SQL
        """

        # Fields
        fields = []
        for column in columns:
            if column.name == autoincrement_column:
                continue
            field_type = self.restore_type(column.type)
            field = {'name': column.name, 'type': field_type}
            if not column.nullable:
                field['constraints'] = {'required': True}
            fields.append(field)

        # Primary key
        pk = []
        for constraint in constraints:
            if isinstance(constraint, sa.PrimaryKeyConstraint):
                for column in constraint.columns:
                    if column.name == autoincrement_column:
                        continue
                    pk.append(column.name)

        # Foreign keys
        fks = []
        if self.__dialect == 'postgresql':
            for constraint in constraints:
                if isinstance(constraint, sa.ForeignKeyConstraint):
                    resource = ''
                    own_fields = []
                    foreign_fields = []
                    for element in constraint.elements:
                        own_fields.append(element.parent.name)
                        if element.column.table.name != table_name:
                            resource = self.restore_bucket(element.column.table.name)
                        foreign_fields.append(element.column.name)
                    if len(own_fields) == len(foreign_fields) == 1:
                        own_fields = own_fields.pop()
                        foreign_fields = foreign_fields.pop()
                    fks.append({
                        'fields': own_fields,
                        'reference': {'resource': resource, 'fields': foreign_fields},
                    })

        # Desscriptor
        descriptor = {}
        descriptor['fields'] = fields
        if len(pk) > 0:
            if len(pk) == 1:
                pk = pk.pop()
            descriptor['primaryKey'] = pk
        if len(fks) > 0:
            descriptor['foreignKeys'] = fks

        return descriptor