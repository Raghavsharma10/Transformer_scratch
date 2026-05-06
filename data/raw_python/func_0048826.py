def set_discrimination_value(self, discrimination):
        """stub"""
        if not isinstance(discrimination, float):
            raise InvalidArgument('discrimination value must be a decimal')
        self.add_decimal_value(discrimination, 'discrimination')