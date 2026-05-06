def run(self, allow_interactive=True):
        """ Enter the native GUI event loop.

        Parameters
        ----------
        allow_interactive : bool
            Is the application allowed to handle interactive mode for console
            terminals?  By default, typing ``python -i main.py`` results in
            an interactive shell that also regularly calls the VisPy event
            loop.  In this specific case, the run() function will terminate
            immediately and rely on the interpreter's input loop to be run
            after script execution.
        """

        if allow_interactive and self.is_interactive():
            inputhook.set_interactive(enabled=True, app=self)
        else:
            return self._backend._vispy_run()