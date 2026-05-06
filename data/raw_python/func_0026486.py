def cli_reload(self, event):
        """Experimental call to reload the component tree"""

        self.log('Reloading all components.')

        self.update_components(forcereload=True)
        initialize()

        from hfos.debugger import cli_compgraph
        self.fireEvent(cli_compgraph())