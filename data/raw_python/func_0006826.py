def compiled_update_func(self):
        """Returns compiled update function"""

        def get_not_none_col_assignment(column_name):
            return ALCHEMY_TEMPLATES.not_none_col_assignment.safe_substitute(col_name=column_name)

        def get_compiled_args(arg_name):
            return ALCHEMY_TEMPLATES.func_arg.safe_substitute(arg_name=arg_name)

        join_string = "\n" + self.tab + self.tab
        columns = [n for n in self.columns if n not in self.primary_keys]
        not_none_col_assignments = join_string.join([get_not_none_col_assignment(n) for n in columns])
        update_args = ", ".join(get_compiled_args(n) for n in columns)
        return ALCHEMY_TEMPLATES.update_function.safe_substitute(not_none_col_assignments=not_none_col_assignments,
                                                                 update_args=update_args,
                                                                 class_name=self.class_name)