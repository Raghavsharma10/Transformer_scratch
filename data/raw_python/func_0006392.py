def deselect_module(self):
        ''' Deselect module and cleanup.
        '''
        self._enabled_fe_channels = []  # ignore any RX sync errors
        self._readout_fifos = []
        self._filter = []
        self._converter = []
        self.dut['TX']['OUTPUT_ENABLE'] = 0
        self._current_module_handle = None
        if isinstance(current_thread(), _MainThread):
            current_thread().name = "MainThread"