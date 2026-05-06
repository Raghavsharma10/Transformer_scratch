def _get_log_model_class(self):
        """Cache for fetching the actual log model object once django is loaded.

        Otherwise, import conflict occur: WorkflowEnabled imports <log_model>
        which tries to import all models to retrieve the proper model class.
        """
        if self.log_model_class is not None:
            return self.log_model_class

        app_label, model_label = self.log_model.rsplit('.', 1)
        self.log_model_class = apps.get_model(app_label, model_label)
        return self.log_model_class