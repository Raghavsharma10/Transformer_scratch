def connected_svcports(self):
        '''The list of all service ports belonging to this component that are
        connected to one or more other ports.

        '''
        return [p for p in self.ports \
                if p.__class__.__name__ == 'CorbaPort' and p.is_connected]