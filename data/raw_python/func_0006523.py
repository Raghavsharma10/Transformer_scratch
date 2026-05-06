def execute(self):
      """ This function Executes the program with set arguments. """
      prog_cmd = self.get_cmd().strip()
      if prog_cmd == '':
         self.status = 'Failure'
         debug.log("Error: No program to execute for %s!"%self.name)
         debug.log(("Could not combine path and arguments into cmdline:"
                    "\n%s %s)\n")%(self.path, ' '.join(self.args)))
      else:
         debug.log("\n\nExecute %s...\n%s" % (self.name, prog_cmd))
         # Create shell script
         script = '%s.sh'%self.name
         if self.wdir != '':
            script = '%s/%s'%(self.wdir, script)
         else:
            script = '%s/%s'%(os.getcwd(), script)
         with open_(script, 'w') as f:
            f.write('#!/bin/bash\n')
            if self.wdir != '':
               f.write('cd {workdir}\n'.format(workdir=self.wdir))
            f.write(
               ('touch {stdout} {stderr}\n'
                'chmod a+r {stdout} {stderr}\n'
                '{cmd} 1> {stdout} 2> {stderr}\n'
                'ec=$?\n').format(
                  stdout=self.stdout,
                  stderr=self.stderr,
                  cmd=prog_cmd
                  )
               )
            if not self.forcewait:
               f.write(('if [ "$ec" -ne "0" ]; then echo "Error" >> {stderr}; '
                        'else echo "Done" >> {stderr}; fi\n').format(
                  stderr=self.stderr))
            f.write('exit $ec\n')
         os.chmod(script, 0o744)
         
         if self.queue is not None:
            # Setup execution of shell script through TORQUE
            other_args = ''
            if self.forcewait: other_args += "-K " # ADDING -K argument if wait() is forced
            # QSUB INFO :: run_time_limit(walltime, dd:hh:mm:ss),
            #              memory(mem, up to 100GB *gigabyte),
            #              processors(ppn, up to 16) # USE AS LITTLE AS NEEDED!
            cmd = ('/usr/bin/qsub '
                   '-l nodes=1:ppn={procs},walltime={hours}:00:00,mem={mem}g '
                   '-r y {workdir_arg} {other_args} {cmd}').format(
                     procs=self.procs,
                     hours=self.walltime,
                     mem=self.mem,
                     workdir_arg="-d %s"%(self.wdir) if self.wdir != '' else '',
                     other_args=other_args,
                     cmd=script)
            debug.log("\n\nTORQUE SETUP %s...\n%s\n" % (self.name, cmd))
         else:
            cmd = script
        
         if self.server is not None:
            cmd = "ssh {server} {cmd}".format(
               server=self.server,
               cmd=quote(cmd)
               )
         self.status = 'Executing'
         # EXECUTING PROGRAM
         self.update_timer(-time()) # TIME START
         if self.forcewait:
            self.p = Popen(cmd)
            ec = self.p.wait()
            if ec == 0:
               debug.log("Program finished successfully!")
               self.status = 'Done'
            else:
               debug.log("Program failed on execution!")
               self.status = 'Failure'
            self.p = None
         else: # WaitOn should be called to determine if the program has ended
            debug.log("CMD: %s"%cmd)
            self.p = Popen(cmd) # shell=True, executable="/bin/bash"
         self.update_timer(time()) # TIME END
         debug.log("timed: %s" % (self.get_time()))