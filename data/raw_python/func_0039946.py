def command_list(self):
        '''returns the 3dDeconvolve command as a list

        The list returned can be run by passing it into a subprocess-like command
        (e.g., neural.run())
        '''
        cmd = ['3dDeconvolve']

        cmd += ['-jobs',multiprocessing.cpu_count()]
        cmd += self.opts
        if(len(self.input_dsets)):
            cmd += ['-input'] + ['%s[%d..%d]' % (dset,self.partial[dset][0],self.partial[dset][1]) if dset in self.partial else dset for dset in self.input_dsets]
        else:
            cmd += ['-nodata']
            if self.reps:
                cmd += [str(self.reps)]
                if self.TR:
                    cmd += [str(self.TR)]
        if self.censor_file:
            censor_file = self.censor_file
            if self.partial:
                # This assumes only one dataset!
                with open(censor_file) as inf:
                    censor = inf.read().split()[self.partial.values()[0][0]:self.partial.values()[0][1]+1]
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        f.write('\n'.join(censor))
                        censor_file = f.name
                        self._del_files.append(f.name)
            cmd += ['-censor', censor_file]
        nfirst = self.nfirst
        if len(self.input_dsets) and self.input_dsets[0] in self.partial:
            nfirst -= self.partial[self.input_dsets[0]][0]
        if nfirst<0:
            nfirst = 0
        cmd += ['-nfirst',str(nfirst)]
        if self.mask:
            if self.mask=='auto':
                cmd += ['-automask']
            else:
                cmd += ['-mask',self.mask]
        cmd += ['-polort',str(self.polort)]

        stim_num = 1

        all_stims = list(self.decon_stims)
        all_stims += [DeconStim(stim,column_file=self.stim_files[stim],base=(stim in self.stim_base)) for stim in self.stim_files]
        for stim in self.stim_times:
            decon_stim = DeconStim(stim,times_file=self.stim_times[stim])
            decon_stim.times_model = self.models[stim] if stim in self.models else self.model_default
            decon_stim.AM1 = (stim in self.stim_am1)
            decon_stim.AM2 = (stim in self.stim_am2)
            decon_stim.base = (stim in self.stim_base)
            all_stims.append(decon_stim)

        if self.partial:
            for i in xrange(len(self.input_dsets)):
                if self.input_dsets[i] in self.partial:
                    new_stims = []
                    for stim in all_stims:
                        stim = stim.partial(self.partial[self.input_dsets[i]][0],self.partial[self.input_dsets[i]][1],i)
                        if stim:
                            new_stims.append(stim)
                    all_stims = new_stims

        cmd += ['-num_stimts',len(all_stims)]

        stimautoname = lambda d,s: 'stimfile_auto-%d-%s_' % (d,s.name) + str(datetime.datetime.now()).replace(" ","_").replace(":",".")

        for stim in all_stims:
            column_file = stim.column_file
            if stim.column!=None:
                column_file = stimautoname(stim_num,stim)
                with open(column_file,"w") as f:
                    f.write('\n'.join([str(x) for x in stim.column]))
            if column_file:
                cmd += ['-stim_file',stim_num,column_file,'-stim_label',stim_num,stim.name]
                if stim.base:
                    cmd += ['-stim_base',stim_num]
                stim_num += 1
                continue
            times_file = stim.times_file
            if stim.times!=None:
                times = list(stim.times)
                if '__iter__' not in dir(times[0]):
                    # a single list
                    times = [times]
                times_file = stimautoname(stim_num,stim)
                with open(times_file,"w") as f:
                    f.write('\n'.join([' '.join([str(x) for x in y]) if len(y)>0 else '*' for y in times]))
            if times_file:
                opt = '-stim_times'
                if stim.AM1:
                    opt = '-stim_times_AM1'
                if stim.AM2:
                    opt = '-stim_times_AM2'
                cmd += [opt,stim_num,times_file,stim.times_model]
                cmd += ['-stim_label',stim_num,stim.name]
                if stim.base:
                    cmd += ['-stim_base',stim_num]
                stim_num += 1

        strip_number = r'[-+]?(\d+)?\*?(\w+)(\[.*?\])?'
        all_glts = {}
        stim_names = [stim.name for stim in all_stims]
        if self.validate_glts:
            for glt in self.glts:
                ok = True
                for stim in self.glts[glt].split():
                    m = re.match(strip_number,stim)
                    if m:
                        stim = m.group(2)
                    if stim not in stim_names:
                        ok = False
                if ok:
                    all_glts[glt] = self.glts[glt]
        else:
            all_glts = self.glts

        cmd += ['-num_glt',len(all_glts)]

        glt_num = 1
        for glt in all_glts:
            cmd += ['-gltsym','SYM: %s' % all_glts[glt],'-glt_label',glt_num,glt]
            glt_num += 1

        if self.bout:
            cmd += ['-bout']
        if self.tout:
            cmd += ['-tout']
        if self.vout:
            cmd += ['-vout']
        if self.rout:
            cmd += ['-rout']

        if self.errts:
            cmd += ['-errts', self.errts]

        if self.prefix:
            cmd += ['-bucket', self.prefix]

        return [str(x) for x in cmd]