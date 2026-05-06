def _produce_return(self, cursor):
        """ Calls callback once with generator.
            :rtype: None
        """
        self.callback(self._row_generator(cursor), *self.cb_args)
        return None