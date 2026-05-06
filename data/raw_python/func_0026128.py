def unlearnValue(self):
        """Unlearn a parameter value by setting it back to its default"""
        defaultValue = self.defaultParamInfo.get(field = "p_filename",
                            native = 0, prompt = 0)
        self.choice.set(defaultValue)