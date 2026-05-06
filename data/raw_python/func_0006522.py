def append_args(self, arg):
      """ This function appends the provided arguments to the program object.
      """
      debug.log("Adding Arguments: %s"%(arg))
      if isinstance(arg, (int,float)): self.args.append(str(arg))
      if isinstance(arg, str): self.args.append(arg)
      if isinstance(arg, list):
         if sys.version_info < (3, 0):
            self.args.extend([str(x) if not isinstance(x, (unicode)) else x.encode('utf-8') for x in arg])
         else:
            self.args.extend([str(x) for x in arg])