def representation_function_compiler(self, func_name):
        """Generic function can be used to compile __repr__ or __unicode__ or __str__"""

        def get_col_accessor(col):
            return ALCHEMY_TEMPLATES.col_accessor.safe_substitute(col=col)

        def get_col_evaluator(col):
            return ALCHEMY_TEMPLATES.col_evaluator.safe_substitute(col=col)

        col_evaluators = ", ".join([get_col_evaluator(n) for n in self.primary_keys])
        col_accessors = ", ".join([get_col_accessor(n) for n in self.primary_keys])

        return ALCHEMY_TEMPLATES.representor_function.safe_substitute(func_name=func_name,
                                                                      col_accessors=col_accessors,
                                                                      col_evaluators=col_evaluators,
                                                                      class_name=self.class_name)