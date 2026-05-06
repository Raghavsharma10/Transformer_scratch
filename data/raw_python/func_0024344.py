def encrypt_password(self):
        """ encrypt password if not already encrypted """
        if self.password and not self.password.startswith('$pbkdf2'):
            self.set_password(self.password)