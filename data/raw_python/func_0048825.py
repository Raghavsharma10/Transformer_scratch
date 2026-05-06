def set_difficulty_value(self, difficulty):
        """stub"""
        if not isinstance(difficulty, float):
            raise InvalidArgument('difficulty value must be a decimal')
        self.add_decimal_value(difficulty, 'difficulty')