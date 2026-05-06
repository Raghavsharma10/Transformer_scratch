def is_current_manager_equals_to(cls, pm):
        """Returns True if this package manager is usable, False otherwise."""
        if hasattr(cls, 'works_result'):
            return cls.works_result
        is_ok = bool(cls._try_get_current_manager() == pm)
        setattr(cls, 'works_result', is_ok)
        return is_ok