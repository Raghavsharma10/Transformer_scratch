def set_password(cls, instance, raw_password):
        """
        sets new password on a user using password manager

        :param instance:
        :param raw_password:
        :return:
        """
        # support API for both passlib 1.x and 2.x
        hash_callable = getattr(
            instance.passwordmanager, "hash", instance.passwordmanager.encrypt
        )
        password = hash_callable(raw_password)
        if six.PY2:
            instance.user_password = password.decode("utf8")
        else:
            instance.user_password = password
        cls.regenerate_security_code(instance)