def _suffix(self):
        """Return appropriate suffix for alignment file"""
        _output_formats={'GCG':'.msf',
                        'GDE':'.gde',
                        'PHYLIP':'.phy',
                        'PIR':'.pir',
                        'NEXUS':'.nxs'}

        if self.Parameters['-output'].isOn():
            return _output_formats[self.Parameters['-output'].Value]
        else:
            return '.aln'