def ready(self, source):
        """All components have initialized, set up the component
        configuration schema-store, run the local server and drop privileges"""

        from hfos.database import configschemastore
        configschemastore[self.name] = self.configschema

        self._start_server()

        if not self.insecure:
            self._drop_privileges()

        self.fireEvent(cli_register_event('components', cli_components))
        self.fireEvent(cli_register_event('drop_privileges', cli_drop_privileges))
        self.fireEvent(cli_register_event('reload_db', cli_reload_db))
        self.fireEvent(cli_register_event('reload', cli_reload))
        self.fireEvent(cli_register_event('quit', cli_quit))
        self.fireEvent(cli_register_event('info', cli_info))