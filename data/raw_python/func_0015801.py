def ready(self):
        """
        Finalizes application configuration.
        """
        import wagtailplus.wagtailrelations.signals.handlers

        self.add_relationship_panels()
        self.add_relationship_methods()
        super(WagtailRelationsAppConfig, self).ready()