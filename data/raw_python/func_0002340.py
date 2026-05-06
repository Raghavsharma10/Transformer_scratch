def _import_plugins(self):
        """
        Internal function, ensure all plugin packages are imported.
        """
        if self.detected:
            return

        # In some cases, plugin scanning may start during a request.
        # Make sure there is only one thread scanning for plugins.
        self.scanLock.acquire()
        if self.detected:
            return  # previous threaded released + completed

        try:
            import_apps_submodule("content_plugins")
            self.detected = True
        finally:
            self.scanLock.release()