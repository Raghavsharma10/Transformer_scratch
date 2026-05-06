def set_inputhook(self, callback):
        """Set PyOS_InputHook to callback and return the previous one."""
        # On platforms with 'readline' support, it's all too likely to
        # have a KeyboardInterrupt signal delivered *even before* an
        # initial ``try:`` clause in the callback can be executed, so
        # we need to disable CTRL+C in this situation.
        ignore_CTRL_C()
        self._callback = callback
        self._callback_pyfunctype = self.PYFUNC(callback)
        pyos_inputhook_ptr = self.get_pyos_inputhook()
        original = self.get_pyos_inputhook_as_func()
        pyos_inputhook_ptr.value = \
            ctypes.cast(self._callback_pyfunctype, ctypes.c_void_p).value
        self._installed = True
        return original