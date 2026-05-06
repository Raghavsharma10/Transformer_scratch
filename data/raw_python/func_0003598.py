def p_statement(self, p):
        """statement : OPTION_AND_VALUE
        """
        p[0] = ['statement', p[1][0], p[1][1]]

        if self.options.get('lowercasenames'):
            p[0][1] = p[0][1].lower()

        if (not self.options.get('nostripvalues') and
                not hasattr(p[0][2], 'is_single_quoted') and
                not hasattr(p[0][2], 'is_double_quoted')):
            p[0][2] = p[0][2].rstrip()