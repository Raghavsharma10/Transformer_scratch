def _setup_converter_graph(self, converter_list, prune_converters):
        '''
        Set up directed conversion graph, pruning unavailable converters as
        necessary
        '''
        for converter in converter_list:
            if prune_converters:
                try:
                    converter.configure()
                except ConverterUnavailable as e:
                    log.warning('%s unavailable: %s' %
                                (converter.__class__.__name__, str(e)))
                    continue

            for in_ in converter.inputs:
                for out in converter.outputs:
                    self.dgraph.add_edge(in_, out, converter.cost)
                    self.converters[(in_, out)] = converter

            if hasattr(converter, 'direct_outputs'):
                self._setup_direct_converter(converter)