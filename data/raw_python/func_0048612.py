def get_solution(self, parameters=None):
        """stub"""
        if not self.has_solution():
            raise IllegalState()
        try:
            if not self.get_text('python_script'):
                return self.get_text('solution').text
            if not parameters:
                parameters = self.get_parameters()
            return self._get_parameterized_text(parameters)
        except Exception:
            return self.get_text('solution').text