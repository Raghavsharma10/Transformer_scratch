def get_xproblem(self, parameters=None):
        """stub"""
        if not self.get_text('python_script'):
            return self.get_text('edxml').text
        if not parameters:
            parameters = self.get_parameters()
        return self._get_parameterized_text(parameters)