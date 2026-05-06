def _get_result_paths(self, data):
        """ Build the dict of result filepaths
        """
        # access data through self.Parameters so we know it's been cast
        # to a FilePath
        wd = self.WorkingDir
        db_name = self.Parameters['-n'].Value
        log_name = self.Parameters['-l'].Value
        result = {}
        result['log'] = ResultPath(Path=wd + log_name, IsWritten=True)
        if self.Parameters['-p'].Value == 'F':
            extensions = ['nhr', 'nin', 'nsq', 'nsd', 'nsi']
        else:
            extensions = ['phr', 'pin', 'psq', 'psd', 'psi']
        for extension in extensions:
            for file_path in glob(wd + (db_name + '*' + extension)):
                # this will match e.g. nr.01.psd and nr.psd
                key = file_path.split(db_name + '.')[1]
                result_path = ResultPath(Path=file_path, IsWritten=True)
                result[key] = result_path
        return result