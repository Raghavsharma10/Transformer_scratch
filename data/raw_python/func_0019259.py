def iofunctions(self):
        """Input/output functions of the model class."""
        lines = Lines()
        for func in ('open_files', 'close_files', 'load_data', 'save_data'):
            if ((func == 'load_data') and
                    (getattr(self.model.sequences, 'inputs', None) is None)):
                continue
            if ((func == 'save_data') and
                ((getattr(self.model.sequences, 'fluxes', None) is None) and
                 (getattr(self.model.sequences, 'states', None) is None))):
                continue
            print('            . %s' % func)
            nogil = func in ('load_data', 'save_data')
            idx_as_arg = func == 'save_data'
            lines.add(1, method_header(
                func, nogil=nogil, idx_as_arg=idx_as_arg))
            for subseqs in self.model.sequences:
                if func == 'load_data':
                    applyfuncs = ('inputs',)
                elif func == 'save_data':
                    applyfuncs = ('fluxes', 'states')
                else:
                    applyfuncs = ('inputs', 'fluxes', 'states')
                if subseqs.name in applyfuncs:
                    if func == 'close_files':
                        lines.add(2, 'self.sequences.%s.%s()'
                                     % (subseqs.name, func))
                    else:
                        lines.add(2, 'self.sequences.%s.%s(self.idx_sim)'
                                     % (subseqs.name, func))
        return lines