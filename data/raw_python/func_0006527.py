def find_err_pattern(self, pattern):
      """ This function will read the standard error of the program and return
          a matching pattern if found.

          EG. prog_obj.FindErrPattern("Update of mySQL failed")
      """
      if self.wdir != '':
         stderr = "%s/%s"%(self.wdir, self.stderr)
      else:
         stderr = self.stderr
      response = []
      # First we check if the file we want to print does exists
      if os.path.exists(stderr):
         with open_(stderr, 'r') as f:
            for line in f:
               if pattern in line:
                  response.append(line.strip())
      else: # FILE DOESN'T EXIST
         debug.log("Error: The stderr file %s does not exist!"%(stderr))
      return response