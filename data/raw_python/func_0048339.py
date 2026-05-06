def p_variable_declaration_noin(self, p):
        """variable_declaration_noin : identifier
                                     | identifier initializer_noin
        """
        if len(p) == 2:
            p[0] = ast.VarDecl(p[1])
        else:
            p[0] = ast.VarDecl(p[1], p[2])