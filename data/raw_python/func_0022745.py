def enable_gui(self, gui=None, app=None):
        """Switch amongst GUI input hooks by name.

        This is a higher level method than :meth:`set_inputhook` - it uses the
        GUI name to look up a registered object which enables the input hook
        for that GUI.

        Parameters
        ----------
        gui : optional, string or None
          If None (or 'none'), clears input hook, otherwise it must be one
          of the recognized GUI names (see ``GUI_*`` constants in module).

        app : optional, existing application object.
          For toolkits that have the concept of a global app, you can supply an
          existing one.  If not given, the toolkit will be probed for one, and if
          none is found, a new one will be created.  Note that GTK does not have
          this concept, and passing an app if ``gui=="GTK"`` will raise an error.

        Returns
        -------
        The output of the underlying gui switch routine, typically the actual
        PyOS_InputHook wrapper object or the GUI toolkit app created, if there was
        one.
        """
        if gui in (None, GUI_NONE):
            return self.disable_gui()
        
        if gui in self.aliases:
            return self.enable_gui(self.aliases[gui], app)
        
        try:
            gui_hook = self.guihooks[gui]
        except KeyError:
            e = "Invalid GUI request {!r}, valid ones are: {}"
            raise ValueError(e.format(gui, ', '.join(self.guihooks)))
        self._current_gui = gui

        app = gui_hook.enable(app)
        if app is not None:
            app._in_event_loop = True
            self.apps[gui] = app        
        return app