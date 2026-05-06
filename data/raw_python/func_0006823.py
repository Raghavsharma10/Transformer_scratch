def compiled_relationships(self):
        """Returns compiled relationship definitions"""

        def get_column_args(column):
            tmp = []
            for arg_name, arg_val in column.items():
                if arg_name not in ('name', 'type', 'reference', 'class'):
                    if arg_name in ('back_populates', ):
                        arg_val = "'{}'".format(arg_val)
                    tmp.append(ALCHEMY_TEMPLATES.column_arg.safe_substitute(arg_name=arg_name,
                                                                            arg_val=arg_val))
            return ", ".join(tmp)

        res = []
        for column in self.relationship_definitions:
            column_args = get_column_args(column)
            column_name = column.get('name')
            cls_name = column.get("class")
            res.append(
                ALCHEMY_TEMPLATES.relationship.safe_substitute(column_name=column_name,
                                                               column_args=column_args,
                                                               class_name=cls_name))
        join_string = "\n" + self.tab
        return join_string.join(res)