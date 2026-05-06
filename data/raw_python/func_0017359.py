def log_transition(self, transition, from_state, instance, *args, **kwargs):
        """Log a transition.

        Args:
            transition (Transition): the name of the performed transition
            from_state (State): the source state
            instance (object): the modified object

        Kwargs:
            Any passed when calling the transition
        """
        logger = logging.getLogger('xworkflows.transitions')
        try:
            instance_repr = u(repr(instance), 'ignore')
        except (UnicodeEncodeError, UnicodeDecodeError):
            instance_repr = u("<bad repr>")
        logger.info(
            u("%s performed transition %s.%s (%s -> %s)"), instance_repr,
            self.__class__.__name__, transition.name, from_state.name,
            transition.target.name)