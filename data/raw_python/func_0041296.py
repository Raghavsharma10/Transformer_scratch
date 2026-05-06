def env_check(self, *args, **kwargs):
        """
        This method provides a common entry for any checks on the 
        environment (input / output dirs, etc)
        """
        b_status    = True
        str_error   = ''
        if not len(self.str_outputDir): 
            b_status = False
            str_error   = 'output directory not specified.'
            self.dp.qprint(str_error, comms = 'error')
            error.warn(self, 'outputDirFail', drawBox = True)
        return {
            'status':       b_status,
            'str_error':    str_error
        }