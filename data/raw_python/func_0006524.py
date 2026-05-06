def wait(self, pattern='Done', interval=None,
              epatterns=['error','Error','STACK','Traceback']):
      """ This function will wait on a given pattern being shown on the last
          line of a given outputfile.

      OPTIONS
         pattern        - The string pattern to recognise when a program
                          finished properly.
         interval       - The amount of seconds to wait between checking the
                          log file.
         epatterns      - A list of string patterns to recognise when a program
                          has finished with an error.
      """
      increasing_interval = False
      if interval is None:
         increasing_interval = True
         interval = 10
      if self.wdir != '':
         stderr = "%s/%s"%(self.wdir, self.stderr)
      else:
         stderr = self.stderr
      debug.log("\nWaiting for %s to finish..."%str(self.name))
      if self.status == 'Executing':
         self.update_timer(-time()) # TIME START
         found = False
         if self.queue is not None:
            # Handling programs running on the compute servers
            # Waiting for error log to be created.
            # Prolonged waiting can be caused by the queue being full, or the
            # server being unavailable.
            debug.log("   Waiting for the error log to be created (%s)..."%(
                     stderr))
            # Set maximum amount of seconds to wait on the errorlog creation,
            # before assuming queue failure.
            max_queued_time = 10800
            while ( not os.path.exists(stderr)
                  and time()+self.timer < max_queued_time
                  and time()+self.timer > 0
                  ):
               debug.log("      Waiting... (max wait time left: %s seconds)"%(
                  str(max_queued_time-time()-self.timer)))
               sleep(interval)
               if increasing_interval:
                  interval *= 1.1
            
            if os.path.exists(stderr):
               if increasing_interval:
                  interval = 10
               # File created looking for pattern
               debug.log('\nError log created, waiting for program to finish...')
            # calculate max loops left based on set walltime and check interval
               max_time = time() + self.walltime * 60 * 60
               while time() < max_time:
                  with open_(stderr) as f:
                     for l in f.readlines()[-5:]: # last five lines
                        if pattern in l:
                           found = True
                           max_time = 0
                           break
                        elif any([ep in l for ep in epatterns]):
                           found = False
                           max_time = 0
                           break
                  if max_time > 0:
                     debug.log('      Waiting... (max wait-time left: %s seconds)'%(
                              str(max_time-time())))
                     sleep(interval)
               if found:
                  debug.log("   Program finished successfully!")
                  self.status = 'Done'
               else:
                  debug.log("Error: Program took too long, or finished with error!")
                  if self.verbose:
                     debug.print_out(
                        "Technical error occurred!\n",
                        "The service was not able to produce a result.\n",
                        ("Please check your settings are correct, and the file "
                        "type matches what you specified.\n"),
                        ("Try again, and if the problem persists please notify the"
                        " technical support.\n")
                        )
                  self.status = 'Failure'
            else:
               debug.log(
                  ("Error: %s still does not exist!\n")%(stderr),
                  ("This error might be caused by the cgebase not being "
                   "available!")
                  )
               if self.verbose:
                  debug.print_out(
                     "Technical error occurred!\n",
                     ("This error might be caused by the server not being "
                     "available!\n"),
                     ("Try again later, and if the problem persists please notify "
                     "the technical support.\n"),
                     "Sorry for any inconvenience.\n"
                     )
               self.status = 'Failure'
            if not self.p is None:
               self.p.wait()
               self.p = None
         else:
            # Handling wrappers running on the webserver
            if self.p is None:
               debug.log("Program not instanciated!")
               self.status = 'Failure'
            else:
               ec = self.p.wait()
               if ec != 0:
                  debug.log("Program failed on execution!")
                  self.status = 'Failure'
               elif os.path.exists(stderr):
                  with open_(stderr) as f:
                     for l in f.readlines()[-5:]: # last five lines
                        if pattern in l:
                           found = True
                           break
                        elif any([ep in l for ep in epatterns]):
                           found = False
                           break
                  if found:
                     debug.log("   Program finished successfully!")
                     self.status = 'Done'
                  else:
                     debug.log("Error: Program failed to finish properly!")
                     if self.verbose:
                        debug.print_out("Technical error occurred!\n",
                           "The service was not able to produce a result.\n",
                           "Please check your settings are correct, and the file "+
                           "type matches what you specified.", "Try again, and if "+
                           "the problem persists please notify the technical "+
                           "support.\n")
                     self.status = 'Failure'
               else:
                  debug.log(("Error: %s does not exist!\n")%(stderr),
                     "This error might be caused by the cgebase not being "+
                     "available!")
                  if self.verbose:
                     debug.print_out("Technical error occurred!\n",
                        "This error might be caused by the server not being "+
                        "available!\n", "Try again later, and if the problem "+
                        "persists please notify the technical support.\n",
                        "Sorry for any inconvenience.\n")
                  self.status = 'Failure'
               self.p = None
         self.update_timer(time()) # TIME END
         debug.log("   timed: %s"%(self.get_time()))
      else:
         debug.log("   The check-out of the program has been sorted previously.")