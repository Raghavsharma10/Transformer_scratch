def print_timers(self):
      ''' PRINT EXECUTION TIMES FOR THE LIST OF PROGRAMS '''
      self.timer += time()
      total_time = self.timer
      tmp = '*  %s  *'
      debug.log(
         '',
         '* '*29,
         tmp%(' '*51),
         tmp%('%s  %s  %s'%('Program Name'.ljust(20), 'Status'.ljust(7), 'Execute Time (H:M:S)')),
         tmp%('='*51)
      )
      for name in self.list:
         if self.exists(name):
            timer = getattr(self, name).get_time()
            status = getattr(self, name).get_status()
            self.timer -= timer
            debug.log(tmp%(self.return_timer(name, status, timer)))
         else:
            debug.log(tmp%("%s  %s -- : -- : --"%(name[:20].ljust(20),'  '*8)))
      debug.log(
         tmp%(self.return_timer('Wrapper', '', self.timer)),
         tmp%('='*51),
         tmp%(self.return_timer('Total', '', total_time)),
         tmp%(' '*51),
         '* '*29,
         ''
      )