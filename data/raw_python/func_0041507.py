def is_training_modified(self):
        """ Returns `True` if training data
            was modified since last training.
            Returns `False` otherwise,
            or if using builtin training data.
        """

        last_modified = self.trainer.get_last_modified()
        if last_modified > self.training_timestamp:
            return True
        else:
            return False