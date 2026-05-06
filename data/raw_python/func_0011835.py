def wait_for_unicode_char(self, ignored_chars=None, timeout=0):
        """Returns a str that contains the single character that was pressed.
        This already respects modifier keys and keyboard layouts. If timeout is
        not none and no key is pressed within the specified timeout, None is
        returned. If a key is ingnored_chars it will be ignored. As argument for
        irgnored_chars any object that has a __contains__ method can be used,
        e.g. a string, a set, a list, etc"""
        return  self.listen_until_return(Handler.unicode_char(ignored_chars), 
                                        timeout=timeout)