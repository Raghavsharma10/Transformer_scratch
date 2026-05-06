def disable_gui(self):
        """Disable GUI event loop integration.
        
        If an application was registered, this sets its ``_in_event_loop``
        attribute to False. It then calls :meth:`clear_inputhook`.
        """
        gui = self._current_gui
        if gui in self.apps:
            self.apps[gui]._in_event_loop = False
        return self.clear_inputhook()