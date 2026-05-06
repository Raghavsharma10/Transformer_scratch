def compiled_foreign_keys(self):
        """Returns compiled foreign key definitions"""

        def get_column_args(column):
            tmp = []
            for arg_name, arg_val in column.items():
                if arg_name not in ('name', 'type', 'reference'):
                    if arg_name in ('server_default', 'server_onupdate'):
                        arg_val = '"{}"'.format(arg_val)
                    tmp.append(ALCHEMY_TEMPLATES.column_arg.safe_substitute(arg_name=arg_name,
                                                                            arg_val=arg_val))
            return ", ".join(tmp)

        def get_fkey_args(column):
            table = column['reference']['table']
            column = column['reference']['column']
            return ALCHEMY_TEMPLATES.foreign_key_arg.safe_substitute(reference_table=table, reference_column=column)

        res = []
        for column in self.foreign_key_definitions:
            column_args = get_column_args(column)
            column_type, type_params = ModelCompiler.get_col_type_info(column.get('type'))
            column_name = column.get('name')
            reference = get_fkey_args(column)
            if column_type in MUTABLE_DICT_TYPES:
                column_type = ALCHEMY_TEMPLATES.mutable_dict_type.safe_substitute(type=column_type,
                                                                                  type_params=type_params)
                type_params = ''
            res.append(
                ALCHEMY_TEMPLATES.foreign_key.safe_substitute(column_name=column_name,
                                                              column_type=column_type,
                                                              column_args=column_args,
                                                              foreign_key_args=reference,
                                                              type_params=type_params))
        join_string = "\n" + self.tab
        return join_string.join(res)