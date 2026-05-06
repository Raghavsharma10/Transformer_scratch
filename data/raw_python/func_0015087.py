def run_simple_topology(cls, config, emitters, result_type=NAMEDTUPLE, max_spout_emits=None):
        """Tests a simple topology. "Simple" means there it has no branches
        or cycles. "emitters" is a list of emitters, starting with a spout
        followed by 0 or more bolts that run in a chain."""
        
        # The config is almost always required. The only known reason to pass
        # None is when calling run_simple_topology() multiple times for the
        # same components. This can be useful for testing spout ack() and fail()
        # behavior.
        if config is not None:
            for emitter in emitters:
                emitter.initialize(config, {})

        with cls() as self:
            # Read from the spout.
            spout = emitters[0]
            spout_id = self.emitter_id(spout)
            old_length = -1
            length = len(self.pending[spout_id])
            while length > old_length and (max_spout_emits is None or length < max_spout_emits):
                old_length = length 
                self.activate(spout)
                spout.nextTuple()
                length = len(self.pending[spout_id])

            # For each bolt in the sequence, consume all upstream input.
            for i, bolt in enumerate(emitters[1:]):
                previous = emitters[i]
                self.activate(bolt)
                while len(self.pending[self.emitter_id(previous)]) > 0:
                    bolt.process(self.read(previous))

        def make_storm_tuple(t, emitter):
            return t
        
        def make_python_list(t, emitter):
            return list(t.values)
        
        def make_python_tuple(t, emitter):
            return tuple(t.values)

        def make_named_tuple(t, emitter):
            return self.get_output_type(emitter)(*t.values)

        if result_type == STORM_TUPLE:
            make = make_storm_tuple
        elif result_type == LIST:
            make = make_python_list
        elif result_type == NAMEDTUPLE:
            make = make_named_tuple
        else:
            assert False, 'Invalid result type specified: %s' % result_type

        result_values = \
            [ [ make(t, emitter) for t in self.processed[self.emitter_id(emitter)]] for emitter in emitters[:-1] ] + \
            [ [ make(t, emitters[-1]) for t in self.pending[self.emitter_id(emitters[-1])] ] ]
        return dict((k, v) for k, v in zip(emitters, result_values))