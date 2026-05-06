def delete_key(self, key_id):
        """Delete user key pointed to by ``key_id``.

        :param int key_id: (required), unique id used by Github
        :returns: bool
        """
        key = self.key(key_id)
        if key:
            return key.delete()
        return False