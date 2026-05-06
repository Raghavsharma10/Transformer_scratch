def print_stdout(self):
      """ This function will read the standard out of the program and print it
      """
      # First we check if the file we want to print does exists
      if self.wdir != '':
         stdout = "%s/%s"%(self.wdir, self.stdout)
      else:
         stdout = self.stdout
      if os.path.exists(stdout):
         with open_(stdout, 'r') as f:
            debug.print_out("\n".join([line for line in f]))
      else: # FILE DOESN'T EXIST
         debug.log("Error: The stdout file %s does not exist!"%(stdout))