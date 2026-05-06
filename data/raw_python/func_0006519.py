def return_timer(self, name, status, timer):
      ''' Return a text formatted timer '''
      timer_template = '%s  %s  %s : %s : %9s'
      t = str(timedelta(0, timer)).split(',')[-1].strip().split(':')
      #t = str(timedelta(0, timer)).split(':')
      if len(t) == 4:
         h, m, s = int(t[0])*24 + int(t[1]), int(t[2]), float(t[3])
      elif len(t) == 3: h, m, s = int(t[0]), int(t[1]), float(t[2])
      else: h, m, s = 0, 0, str(t)
      return timer_template%(
         name[:20].ljust(20),
         status[:7].ljust(7),
         '%3d'%h if h != 0 else ' --',
         '%2d'%m if m != 0 else '--',
         '%.6f'%s if isinstance(s, float) else s
      )