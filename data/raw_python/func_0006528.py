def find_out_pattern(self, pattern):
      """ This function will read the standard error of the program and return
          a matching pattern if found.

          EG. prog_obj.FindErrPattern("Update of mySQL failed")
      """
      if self.wdir != '':
         stdout = "%s/%s"%(self.wdir, self.stdout)
      else:
         stdout = self.stdout
      response = []
      # First we check if the file we want to print does exists
      if os.path.exists(stdout):
         with open_(stdout, 'r') as f:
            for line in f:
               if pattern in line:
                  response.append(line.strip())
      else: # FILE DOESN'T EXIST
         debug.log("Error: The stdout file %s does not exist!"%(stdout))
      return response