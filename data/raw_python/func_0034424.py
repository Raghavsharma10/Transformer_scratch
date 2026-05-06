def remove_tag(self, tag, value, silent_fail=False):
        """ we try to remove supplied pair tag -- value, and if does not exist outcome depends on the silent_fail flag """
        try:
            self.tags.remove((tag, value))
        except ValueError as err:
            if not silent_fail:
                raise err