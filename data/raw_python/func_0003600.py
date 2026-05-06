def p_contents(self, p):
        """contents : contents statements
                    | contents comment
                    | contents include
                    | contents includeoptional
                    | contents block
                    | statements
                    | comment
                    | include
                    | includeoptional
                    | block
        """
        n = len(p)
        if n == 3:
            p[0] = p[1] + [p[2]]
        else:
            p[0] = ['contents', p[1]]