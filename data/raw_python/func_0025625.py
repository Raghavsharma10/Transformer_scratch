def _nucmer_command(self, ref, qry, outprefix):
        '''Construct the nucmer command'''
        if self.use_promer:
            command = 'promer'
        else:
            command = 'nucmer'

        command += ' -p ' + outprefix

        if self.breaklen is not None:
            command += ' -b ' + str(self.breaklen)

        if self.diagdiff is not None and not self.use_promer:
            command += ' -D ' + str(self.diagdiff)

        if self.diagfactor:
            command += ' -d ' + str(self.diagfactor)

        if self.maxgap:
            command += ' -g ' + str(self.maxgap)

        if self.maxmatch:
            command += ' --maxmatch'

        if self.mincluster is not None:
            command += ' -c ' + str(self.mincluster)

        if not self.simplify and not self.use_promer:
        	command += ' --nosimplify'

        return command + ' ' + ref + ' ' + qry