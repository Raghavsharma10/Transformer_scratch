def set_pseudo_guessing_value(self, pseudo_guessing):
        """stub"""
        if not isinstance(pseudo_guessing, float):
            raise InvalidArgument('pseudo-guessing value must be a decimal')
        self.add_decimal_value(pseudo_guessing, 'pseudoGuessing')