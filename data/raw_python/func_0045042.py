def add(self, name, nestable, **kw):
        """
        Adds a level to the nesting and creates a checkpoint that can be
        reverted to later for aggregation by calling :meth:`SConsWrap.pop`.

        :param name: Identifier for the nest level
        :param nestable: A nestable object - see
            :meth:`Nest.add() <nestly.core.Nest.add>`.
        :param kw: Additional parameters to pass to
            :meth:`Nest.add() <nestly.core.Nest.add>`.
        """
        self.checkpoints[name] = self.nest
        self.nest = copy.copy(self.nest)
        return self.nest.add(name, nestable, **kw)