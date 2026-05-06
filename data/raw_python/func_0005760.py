def _add_new_spawn_method(cls):
    """
    TODO
    """

    def new_spawn_method(self, dependency_mapping):
        # TODO/FIXME: Check that this does the right thing:
        # (i) the spawned generator is independent of the original one (i.e. they can be reset independently without altering the other's behaviour)
        # (ii) ensure that it also works if this custom generator's __init__ requires additional arguments
        #new_instance = self.__class__()
        #
        # FIXME: It would be good to explicitly spawn the field generators of `self`
        #        here because this would ensure that the internal random generators
        #        of the spawned versions are in the same state as the ones in `self`.
        #        This would guarantee that the spawned custom generator produces the
        #        same elements as `self` even before reset() is called explicitly.
        new_instance = cls()
        return new_instance

    cls.spawn = new_spawn_method