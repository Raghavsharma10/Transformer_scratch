def rule(self, key):
        """Decorate as a rule for a key in top level JSON."""
        def register(f):
            self.rules[key] = f
            return f
        return register