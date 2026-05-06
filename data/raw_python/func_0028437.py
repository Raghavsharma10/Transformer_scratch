def init_check(self, check, obj):
        """
        Adds a given check callback with the provided object to the list
        of checks. Useful for built-ins but also advanced custom checks.
        """
        self.logger.info('Adding extension check %s' % check.__name__)
        check = functools.wraps(check)(functools.partial(check, obj))
        self.check(func=check)