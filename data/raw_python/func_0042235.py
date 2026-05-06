def kls_name(self):
        """Determine python name for group"""
        # Determine kls for group
        if not self.parent or not self.parent.name:
            return 'Test{0}'.format(self.name)
        else:
            use = self.parent.kls_name
            if use.startswith('Test'):
                use = use[4:]

            return 'Test{0}_{1}'.format(use, self.name)