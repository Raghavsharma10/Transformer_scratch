def p_case_clauses(self, p):
        """case_clauses : case_clause
                        | case_clauses case_clause
        """
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[1].append(p[2])
            p[0] = p[1]