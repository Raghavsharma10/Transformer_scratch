def replace_variable(self, variable):
        """Substitute variables with numeric values"""
        if variable == 'x':
            return self.value
        if variable == 't':
            return self.timedelta
        raise ValueError("Invalid variable %s", variable)