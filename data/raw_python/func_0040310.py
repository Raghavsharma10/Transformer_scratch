def itervisit(
            self, iterable, gentype=types.GeneratorType,
            exhaust_generators=True):
        '''The main visit function. Visits the passed-in node and calls
        finalize.
        '''
        self.iterable = iter(iterable)
        for token in self.iterable:
            result = self.itervisit_node(token)
            if exhaust_generators and isinstance(result, gentype):
                for output in result:
                    yield output
            elif result is not None:
                yield result
        result = self.finalize()
        if result is self:
            return
        if isinstance(result, gentype):
            for output in result:
                yield output