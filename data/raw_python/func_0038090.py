def bcrypt_check_password(self, raw_password):
    """
    Returns a boolean of whether the *raw_password* was correct.

    Attempts to validate with bcrypt, but falls back to Django's
    ``User.check_password()`` if the hash is incorrect.

    If ``BCRYPT_MIGRATE`` is set, attempts to convert sha1 password to bcrypt
    or converts between different bcrypt rounds values.

    .. note::

        In case of a password migration this method calls ``User.save()`` to
        persist the changes.
    """
    pwd_ok = False
    should_change = False
    if self.password.startswith('bc$'):
        salt_and_hash = self.password[3:]
        pwd_ok = bcrypt.hashpw(smart_str(raw_password), salt_and_hash) == salt_and_hash
        if pwd_ok:
            rounds = int(salt_and_hash.split('$')[2])
            should_change = rounds != get_rounds()
    elif _check_password(self, raw_password):
        pwd_ok = True
        should_change = True

    if pwd_ok and should_change and is_enabled() and migrate_to_bcrypt():
        self.set_password(raw_password)
        salt_and_hash = self.password[3:]
        assert bcrypt.hashpw(raw_password, salt_and_hash) == salt_and_hash
        self.save()

    return pwd_ok