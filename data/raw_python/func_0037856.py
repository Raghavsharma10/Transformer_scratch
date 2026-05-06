def register_persistent_rest_pair(self, persistent_model_class, rest_model_class):
        """
        :param persistent_model_class:
        :param rest_model_class:
        """
        self.register_adapter(ModelAdapter(
            rest_model_class=rest_model_class,
            persistent_model_class=persistent_model_class
        ))