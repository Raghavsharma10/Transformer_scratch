def convert_descriptor(self, bucket, descriptor, index_fields=[], autoincrement=None):
        """Convert descriptor to SQL
        """

        # Prepare
        columns = []
        indexes = []
        fallbacks = []
        constraints = []
        column_mapping = {}
        table_name = self.convert_bucket(bucket)
        table_comment = _get_comment(descriptor.get('title', ''), descriptor.get('description', ''))
        schema = tableschema.Schema(descriptor)

        # Autoincrement
        if autoincrement is not None:
            columns.append(sa.Column(
                autoincrement, sa.Integer, autoincrement=True, nullable=False))

        # Fields
        for field in schema.fields:
            column_type = self.convert_type(field.type)
            if not column_type:
                column_type = sa.Text
                fallbacks.append(field.name)
            nullable = not field.required
            table_comment = _get_field_comment(field)
            unique = field.constraints.get('unique', False)
            column = sa.Column(field.name, column_type, nullable=nullable, comment=table_comment,
                               unique=unique)
            columns.append(column)
            column_mapping[field.name] = column

        # Primary key
        pk = descriptor.get('primaryKey', None)
        if pk is not None:
            if isinstance(pk, six.string_types):
                pk = [pk]
        if autoincrement is not None:
            if pk is not None:
                pk = [autoincrement] + pk
            else:
                pk = [autoincrement]
        if pk is not None:
            constraint = sa.PrimaryKeyConstraint(*pk)
            constraints.append(constraint)

        # Foreign keys
        if self.__dialect == 'postgresql':
            fks = descriptor.get('foreignKeys', [])
            for fk in fks:
                fields = fk['fields']
                resource = fk['reference']['resource']
                foreign_fields = fk['reference']['fields']
                if isinstance(fields, six.string_types):
                    fields = [fields]
                if resource != '':
                    table_name = self.convert_bucket(resource)
                if isinstance(foreign_fields, six.string_types):
                    foreign_fields = [foreign_fields]
                composer = lambda field: '.'.join([table_name, field])
                foreign_fields = list(map(composer, foreign_fields))
                constraint = sa.ForeignKeyConstraint(fields, foreign_fields)
                constraints.append(constraint)

        # Indexes
        if self.__dialect == 'postgresql':
            for index, index_definition in enumerate(index_fields):
                name = table_name + '_ix%03d' % index
                index_columns = [column_mapping[field] for field in index_definition]
                indexes.append(sa.Index(name, *index_columns))

        return columns, constraints, indexes, fallbacks, table_comment