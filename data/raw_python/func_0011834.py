def wait_for_keys_modified(self, *keys, modifiers_to_check=_mod_keys,
                               timeout=0):
        """The same as wait_for_keys, but returns a frozen_set which contains 
        the pressed key, and the modifier keys.

        :param modifiers_to_check: iterable of modifiers for which the function
            will check whether they are pressed

        :type modifiers: Iterable[int]"""

        set_mods = pygame.key.get_mods()
        return frozenset.union(
            frozenset([self.wait_for_keys(*keys, timeout=timeout)]),
            EventListener._contained_modifiers(set_mods, modifiers_to_check))