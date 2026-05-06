def get_cmd(self):
      """ This function combines and return the commanline call of the program.
      """
      cmd = []
      if self.path is not None:
         if '/' in self.path and not os.path.exists(self.path):
            debug.log('Error: path contains / but does not exist: %s'%self.path)
         else:
            if self.ptype is not None:
               if os.path.exists(self.ptype):
                  cmd.append(self.ptype)
               elif '/' not in self.ptype:
                  for path in os.environ["PATH"].split(os.pathsep):
                     path = path.strip('"')
                     ppath = os.path.join(path, self.ptype)
                     if os.path.isfile(ppath):
                        cmd.append(ppath)
                        break
            cmd.append(self.path)
            if sys.version_info < (3, 0):
               cmd.extend([str(x) if not isinstance(x, (unicode)) else x.encode('utf-8') for x in [quote(str(x)) for x in self.args]+self.unquoted_args])
            else:
               cmd.extend([str(x) for x in [quote(str(x)) for x in self.args]+self.unquoted_args])
      else:
         debug.log('Error: Program path not set!')
      return ' '.join(cmd)