def set_option(self, name, val, action=Empty, opts=Empty):
        """Determine which options were specified outside of the defaults"""
        if action is Empty and opts is Empty:
            self.specified.append(name)
            super(SpecRegister, self).set_option(name, val)
        else:
            super(SpecRegister, self).set_option(name, val, action, opts)