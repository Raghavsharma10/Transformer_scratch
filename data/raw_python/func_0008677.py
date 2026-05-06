def db_log(self, transition, from_state, instance, *args, **kwargs):
        """Logs the transition into the database."""
        if self.log_model:
            model_class = self._get_log_model_class()

            extras = {}
            for db_field, transition_arg, default in model_class.EXTRA_LOG_ATTRIBUTES:
                extras[db_field] = kwargs.get(transition_arg, default)

            return model_class.log_transition(
                modified_object=instance,
                transition=transition.name,
                from_state=from_state.name,
                to_state=transition.target.name,
                **extras)