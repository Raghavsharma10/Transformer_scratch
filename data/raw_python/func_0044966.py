def creating_schema_and_index(self, models, func):
        """
        Executes given functions with given models.

        Args:
            models: models to execute
            func: function name to execute

        Returns:

        """
        waiting_models = []
        self.base_thread.do_with_submit(func, models, waiting_models, threads=self.threads)
        if waiting_models:
            print("WAITING MODELS ARE CHECKING...")
            self.creating_schema_and_index(waiting_models, func)