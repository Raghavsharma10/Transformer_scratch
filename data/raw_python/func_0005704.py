def add_new_spawn_method(obj):
    """
    TODO
    """

    def new_spawn(self):
        # TODO/FIXME: Check that this does the right thing:
        # (i) the spawned generator is independent of the original one (i.e. they can be reset independently without altering the other's behaviour)
        # (ii) ensure that it also works if this custom generator's __init__ requires additional arguments
        new_instance = self.__class__()
        return new_instance

    obj._spawn = new_spawn