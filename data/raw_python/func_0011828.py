def unicode_char(ignored_chars=None):
        """returns a handler that listens for unicode characters"""
        return lambda e: e.unicode if e.type == pygame.KEYDOWN \
            and ((ignored_chars is None) 
                  or (e.unicode not in ignored_chars))\
            else EventConsumerInfo.DONT_CARE