def connected_inports(self):
        '''The list of all input ports belonging to this component that are
        connected to one or more other ports.

        '''
        return [p for p in self.ports \
                if p.__class__.__name__ == 'DataInPort' and p.is_connected]