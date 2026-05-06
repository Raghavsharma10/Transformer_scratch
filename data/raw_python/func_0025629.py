def run(self):
        '''
        Change to a temp directory
        Run bash script containing commands
        Place results in specified output file
        Clean up temp directory
        '''
        qry = os.path.abspath(self.qry)
        ref = os.path.abspath(self.ref)
        outfile = os.path.abspath(self.outfile)
        tmpdir = tempfile.mkdtemp(prefix='tmp.run_nucmer.', dir=os.getcwd())
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        script = 'run_nucmer.sh'
        self._write_script(script, ref, qry, outfile)
        syscall.run('bash ' + script, verbose=self.verbose)
        os.chdir(original_dir)
        shutil.rmtree(tmpdir)