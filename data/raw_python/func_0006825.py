def compiled_init_func(self):
        """Returns compiled init function"""

        def get_column_assignment(column_name):
            return ALCHEMY_TEMPLATES.col_assignment.safe_substitute(col_name=column_name)

        def get_compiled_args(arg_name):
            return ALCHEMY_TEMPLATES.func_arg.safe_substitute(arg_name=arg_name)

        join_string = "\n" + self.tab + self.tab
        column_assignments = join_string.join([get_column_assignment(n) for n in self.columns])
        init_args = ", ".join(get_compiled_args(n) for n in self.columns)
        return ALCHEMY_TEMPLATES.init_function.safe_substitute(col_assignments=column_assignments,
                                                               init_args=init_args)