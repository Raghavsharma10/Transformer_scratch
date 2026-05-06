def eval(self, statement, ctxt=None):
        """Eval a single statement (something returnable)
        """
        self._no_debug = True

        statement = statement.strip()

        if not statement.endswith(";"):
            statement += ";"

        ast = self._parse_string(statement, predefines=False)

        self._dlog("evaluating statement: {}".format(statement))
        
        try:
            res = None
            for child in ast.children():
                res = self._handle_node(child, self._scope, self._ctxt, self._stream)
            return res
        except errors.InterpReturn as e:
            return e.value
        finally:
            self._no_debug = False