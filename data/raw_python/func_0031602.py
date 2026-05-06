def install(self, force_install=False):
        """
        Installs an app. Blocks until the installation is complete, or raises :exc:`AppInstallError` if it fails.

        While this method runs, "progress" events will be emitted regularly with the following signature: ::

           (sent_this_interval, sent_total, total_size)

        :param force_install: Install even if installing this pbw on this platform is usually forbidden.
        :type force_install: bool
        """
        if not (force_install or self._bundle.should_permit_install()):
            raise AppInstallError("This pbw is not supported on this platform.")
        if self._pebble.firmware_version.major < 3:
            self._install_legacy2()
        else:
            self._install_modern()