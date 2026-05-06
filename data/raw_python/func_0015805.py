def ready(self):
        """
        Finalizes application configuration.
        """
        self.add_rollback_panels()
        self.add_rollback_methods()
        super(WagtailRollbacksAppConfig, self).ready()