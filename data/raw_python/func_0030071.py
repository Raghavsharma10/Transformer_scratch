def count(self):
        '''
        A count based on `count_field` and `format_args`.
        '''
        args = self.format_args
        if args is None or \
                (isinstance(args, dict) and self.count_field not in args):
            raise TypeError("count is required")
        return args[self.count_field] if isinstance(args, dict) else args