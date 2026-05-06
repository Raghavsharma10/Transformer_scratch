def _basic_iterator(self):
        """this iterator yields individual crash_ids and/or Nones from the
        iterator specified by the "_create_iter" method. Bare values yielded
        by the "_create_iter" method get wrapped into an *args, **kwargs form.
        That form is then used by the task manager as the arguments to the
        worker function."""
        for x in self._create_iter():
            if x is None or isinstance(x, tuple):
                yield x
            else:
                yield ((x,), {})
            self._action_between_each_iteration()
        else:
            # when the iterator is exhausted, yield None as this is an
            # indicator to some of the clients to take an action.
            # This is a moribund action, but in this current refactoring
            # we don't want to change old behavior
            yield None
        self._action_after_iteration_completes()