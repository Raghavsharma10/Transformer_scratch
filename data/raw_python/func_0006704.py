def init_login(self, from_local=False):
        """Display login screen. May ask for local data loading if from_local is True."""
        if self.toolbar:
            self.removeToolBar(self.toolbar)
        widget_login = login.Loading(self.statusBar(), self.theory_main)
        self.centralWidget().addWidget(widget_login)
        widget_login.loaded.connect(self.init_tabs)
        widget_login.canceled.connect(self._quit)
        widget_login.updated.connect(self.on_update_at_launch)
        if from_local:
            widget_login.propose_load_local()
        else:
            self.statusBar().showMessage("Données chargées depuis le serveur.", 5000)