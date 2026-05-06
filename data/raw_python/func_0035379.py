def p_reserved_word(self, p):
        """reserved_word : BREAK
                         | CASE
                         | CATCH
                         | CONTINUE
                         | DEBUGGER
                         | DEFAULT
                         | DELETE
                         | DO
                         | ELSE
                         | FINALLY
                         | FOR
                         | FUNCTION
                         | IF
                         | IN
                         | INSTANCEOF
                         | NEW
                         | RETURN
                         | SWITCH
                         | THIS
                         | THROW
                         | TRY
                         | TYPEOF
                         | VAR
                         | VOID
                         | WHILE
                         | WITH
                         | NULL
                         | TRUE
                         | FALSE
                         | CLASS
                         | CONST
                         | ENUM
                         | EXPORT
                         | EXTENDS
                         | IMPORT
                         | SUPER
        """
        p[0] = self.asttypes.Identifier(p[1])
        p[0].setpos(p)