def check_password(cls, instance, raw_password, enable_hash_migration=True):
        """
        checks string with users password hash using password manager

        :param instance:
        :param raw_password:
        :param enable_hash_migration: if legacy hashes should be migrated
        :return:
        """
        verified, replacement_hash = instance.passwordmanager.verify_and_update(
            raw_password, instance.user_password
        )
        if enable_hash_migration and replacement_hash:
            if six.PY2:
                instance.user_password = replacement_hash.decode("utf8")
            else:
                instance.user_password = replacement_hash
        return verified