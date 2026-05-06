def super_pipe(self):
        """
        Creates a mechanism for us to internally bind two different
        operations together in a shared pipeline on the class.
        This will temporarily set self._pipe to be this new pipeline,
        during this context and then when it leaves the context
        reset self._pipe to its original value.

        Example:
            def get_set(self, key, val)
                with self.super_pipe as pipe:
                    res = self.get(key)
                    self.set(key, val)
                    return res

        This will have the effect of using only one network round trip if no
        pipeline was passed to the constructor.

        This method is still considered experimental and we are working out
        the details, so don't use it unless you feel confident you have a
        legitimate use-case for using this.
        """
        orig_pipe = self._pipe

        def exit_handler():
            self._pipe = orig_pipe

        self._pipe = autoexec(orig_pipe, name=self.connection,
                              exit_handler=exit_handler)

        return self._pipe