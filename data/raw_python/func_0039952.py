def partial(self,start=0,end=None,run=0):
        '''chops the stimulus by only including time points ``start`` through ``end`` (in reps, inclusive; ``None``=until the end)
        if using stim_times-style simulus, will change the ``run``'th run. If a column, will just chop the column'''
        self.read_file()
        decon_stim = copy.copy(self)
        if start<0:
            start = 0
        if self.type()=="column":
            decon_stim.column_file = None
            if end>=len(decon_stim.column):
                end = None
            if end==None:
                decon_stim.column = decon_stim.column[start:]
            else:
                decon_stim.column = decon_stim.column[start:end+1]
            if len(decon_stim.column)==0:
                return None
        if self.type()=="times":
            if self.TR==None:
                nl.notify('Error: cannot get partial segment of a stim_times stimulus without a TR',level=nl.level.error)
                return None
            def time_in(a):
                first_number = r'^(\d+(\.\d+)?)'
                if isinstance(a,basestring):
                    m = re.match(first_number,a)
                    if m:
                        a = m.group(1)
                    else:
                        nl.notify('Warning: cannot intepret a number from the stim_time: "%s"' % a,level=nl.level.warning)
                        return False
                a = float(a)/self.TR
                if a>=start and (end==None or a<=end):
                    return True
                return False

            decon_stim.times_file = None
            if len(decon_stim.times)==0 or '__iter__' not in dir(decon_stim.times[0]):
                decon_stim.times = [decon_stim.times]
            decon_stim.times[run] = [x for x in decon_stim.times[run] if time_in(x)]
            if len(nl.flatten(decon_stim.times))==0:
                return None
        return decon_stim