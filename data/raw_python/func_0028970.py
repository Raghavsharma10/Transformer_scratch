def result(self):
        """Evaluate expression and return result"""
        # Module(body=[Expr(value=...)])
        return self.eval_(ast.parse(self.expr).body[0].value)