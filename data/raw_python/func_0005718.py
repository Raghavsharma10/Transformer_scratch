def spawn(self, spawn_mapping=None):
        """
        Return an exact copy of this generator which behaves the same way
        (i.e., produces the same elements in the same order) but is otherwise
        independent, i.e. there is no link between the two generators
        (as opposed to a cloned generator, which is automatically reset
        whenever the original generator is reset).
        """
        spawn_mapping = spawn_mapping or SpawnMapping()

        if self.parent is not None:
            if self.parent in spawn_mapping:
                # Return new clone of the mapped parent
                return spawn_mapping[self.parent].clone()
            else:
                raise TohuCloneError("Cannot spawn a cloned generator without being able to map its parent.")
        else:
            new_obj = spawn_mapping.spawn_generator(self)
            return new_obj