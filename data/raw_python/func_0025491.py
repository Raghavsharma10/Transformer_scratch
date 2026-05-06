def set_r_value(self, r_var: str, *, notify_changed=True) -> None:
        """Used to signal changes to the ref var, which are kept in document controller. ugh."""
        self.r_var = r_var
        self._description_changed()
        if notify_changed:  # set to False to set the r-value at startup; avoid marking it as a change
            self.__notify_description_changed()