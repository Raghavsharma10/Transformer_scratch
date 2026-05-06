def get_string(string):
   """ This function checks if a path was given as string, and tries to read the
       file and return the string.
   """
   truestring = string
   if string is not None:
      if '/' in string:
         if os.path.isfile(string):
            try:
               with open_(string,'r') as f:
                  truestring = ' '.join(line.strip() for line in f)
            except: pass
      if truestring.strip() == '': truestring = None
   return truestring