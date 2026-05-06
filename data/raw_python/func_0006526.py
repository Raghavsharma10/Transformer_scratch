def find_out_var(self, varnames=[]):
      """ This function will read the standard out of the program, catch
          variables and return the values

          EG. #varname=value
      """
      if self.wdir != '':
         stdout = "%s/%s"%(self.wdir, self.stdout)
      else:
         stdout = self.stdout
      response = [None]*len(varnames)
      # First we check if the file we want to print does exists
      if os.path.exists(stdout):
         with open_(stdout, 'r') as f:
            for line in f:
               if '=' in line:
                  var = line.strip('#').split('=')
                  value = var[1].strip()
                  var = var[0].strip()
                  if var in varnames: response[varnames.index(var)] = value
      else: # FILE DOESN'T EXIST
         debug.log("Error: The stdout file %s does not exist!"%(stdout))
      return response