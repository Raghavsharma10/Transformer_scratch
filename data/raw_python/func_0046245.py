def decorators(self, relation):
        """
        Return prefixes for tuple.

        :param int relation: relation of string value to actual value
        """
        if self.CONFIG.show_approx_str:
            approx_str = Decorators.relation_to_symbol(relation)
        else:
            approx_str = ''

        return _Decorators(approx_str=approx_str)