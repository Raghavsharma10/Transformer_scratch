def _next(self, state_class, *args):
        """Transition into the next state.

        :param type state_class: a subclass of :class:`State`. It is intialized
          with the communication object and :paramref:`args`
        :param args: additional arguments
        """
        self._communication.state = state_class(self._communication, *args)