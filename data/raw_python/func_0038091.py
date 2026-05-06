def bcrypt_set_password(self, raw_password):
    """
    Sets the user's password to *raw_password*, hashed with bcrypt.
    """
    if not is_enabled() or raw_password is None:
        _set_password(self, raw_password)
    else:
        salt = bcrypt.gensalt(get_rounds())
        self.password = 'bc$' + bcrypt.hashpw(smart_str(raw_password), salt)