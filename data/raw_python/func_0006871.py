def print2file(self, logfile, print2screen, addLineFeed, *lst):
      """ This function prints to the screen and logs to a file, all the strings
      given.
      # print2screen eg. True, *lst is a commaseparated list of strings
      """
      if addLineFeed:
         linefeed = '\n'
      else: linefeed = ''
      if print2screen: print(linefeed.join(str(string) for string in lst))
      try: file_instance = isinstance(logfile, file)
      except NameError as e:
         from io import IOBase
         try: file_instance = isinstance(logfile, IOBase)
         except: raise e
      if file_instance:
         logfile.write(linefeed.join(str(string) for string in lst) + linefeed)
      elif isinstance(logfile, str) and os.path.exists(logfile):
         with open_(logfile, 'a') as f:
            f.write(linefeed.join(str(string) for string in lst) + linefeed)
      elif not print2screen: # Print to screen if there is no outputfile
         print(linefeed.join(str(string) for string in lst))