def connected_outports(self):
        '''The list of all output ports belonging to this component that are
        connected to one or more other ports.

        '''
        return [p for p in self.ports \
                    if p.__class__.__name__ == 'DataOutPort' \
                    and p.is_connected]