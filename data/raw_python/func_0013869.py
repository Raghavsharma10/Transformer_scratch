def _assign_funcs(self, by_name=False, inst_module=None):        
        """Assign all external science instrument methods to Instrument object.
        """
        import importlib
        # set defaults
        self._list_rtn = self._pass_func
        self._load_rtn = self._pass_func
        self._default_rtn = self._pass_func
        self._clean_rtn = self._pass_func
        self._init_rtn = self._pass_func
        self._download_rtn = self._pass_func
        # default params
        self.directory_format = None
        self.file_format = None
        self.multi_file_day = False
        self.orbit_info = None
                        
        if by_name: 
            # look for code with filename name, any errors passed up
            inst = importlib.import_module(''.join(('.', self.platform, '_',
                                           self.name)),
                                           package='pysat.instruments')
        elif inst_module is not None:
            # user supplied an object with relevant instrument routines
            inst = inst_module
        else:
            # no module or name info, default pass functions assigned
            return

        try:
            self._load_rtn = inst.load
            self._list_rtn = inst.list_files
            self._download_rtn = inst.download
        except AttributeError:
            estr = 'A load, file_list, and download routine are required for '
            raise AttributeError('{:s}every instrument.'.format(estr))
        try:
            self._default_rtn = inst.default
        except AttributeError:
            pass
        try:
            self._init_rtn = inst.init
        except AttributeError:
            pass   
        try:
            self._clean_rtn = inst.clean
        except AttributeError:
            pass

        # look for instrument default parameters
        try:
            self.directory_format = inst.directory_format
        except AttributeError:
            pass
        try:
            self.multi_file_day = inst.multi_file_day
        except AttributeError:
            pass
        try:
            self.orbit_info = inst.orbit_info
        except AttributeError:
            pass

        return