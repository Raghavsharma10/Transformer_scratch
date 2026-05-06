def _checkAttribs(self, strict):
        """Check initial attributes to make sure they are legal"""
        if self.min:
            warning("Minimum value not allowed for boolean-type parameter " +
                    self.name, strict)
            self.min = None
        if self.max:
            if not self.prompt:
                warning("Maximum value not allowed for boolean-type parameter " +
                                self.name + " (probably missing comma)",
                                strict)
                # try to recover by assuming max string is prompt
                self.prompt = self.max
            else:
                warning("Maximum value not allowed for boolean-type parameter " +
                        self.name, strict)
            self.max = None
        if self.choice:
            warning("Choice values not allowed for boolean-type parameter " +
                    self.name, strict)
            self.choice = None