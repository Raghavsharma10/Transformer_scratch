def set_tolerance_value(self, tolerance):
        """stub"""
        # include index because could be multiple response / tolerance pairs
        if not isinstance(tolerance, float):
            raise InvalidArgument('tolerance value must be a decimal')
        self.add_decimal_value(tolerance, 'tolerance')