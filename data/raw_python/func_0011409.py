def _pfp__set_value(self, new_val):
        """Set the new value if type checking is passes, potentially
        (TODO? reevaluate this) casting the value to something else

        :new_val: The new value
        :returns: TODO

        """
        if self._pfp__frozen:
            raise errors.UnmodifiableConst()
        self._pfp__value = self._pfp__get_root_value(new_val)
        self._pfp__notify_parent()