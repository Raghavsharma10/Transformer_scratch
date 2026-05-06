def p_case_block(self, p):
        """
        case_block \
            : LBRACE case_clauses_opt RBRACE
            | LBRACE case_clauses_opt default_clause case_clauses_opt RBRACE
        """
        statements = []
        for s in p[2:-1]:
            if isinstance(s, list):
                for i in s:
                    statements.append(i)
            elif isinstance(s, self.asttypes.Default):
                statements.append(s)
        p[0] = self.asttypes.CaseBlock(statements)
        p[0].setpos(p)