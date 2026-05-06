def _log(self, monitors, iteration, label='', suffix=''):
        '''Log the state of the optimizer on the console.

        Parameters
        ----------
        monitors : OrderedDict
            A dictionary of monitor names mapped to values. These names and
            values are what is being logged.
        iteration : int
            Optimization iteration that we are logging.
        label : str, optional
            A label for the name of the optimizer creating the log line.
            Defaults to the name of the current class.
        suffix : str, optional
            A suffix to add to the end of the log line, if any.
        '''
        label = label or self.__class__.__name__
        fields = (('{}={:.6f}').format(k, v) for k, v in monitors.items())
        util.log('{} {} {}{}'.format(label, iteration, ' '.join(fields), suffix))