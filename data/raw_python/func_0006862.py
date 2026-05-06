def mkpath(filepath, permissions=0o777):
   """ This function executes a mkdir command for filepath and with permissions
   (octal number with leading 0 or string only)
   # eg. mkpath("path/to/file", "0o775")
   """
   # Converting string of octal to integer, if string is given.
   if isinstance(permissions, str):
      permissions = sum([int(x)*8**i for i,x in enumerate(reversed(permissions))])
   # Creating directory
   if not os.path.exists(filepath):
      debug.log("Creating Directory %s (permissions: %s)"%(
         filepath, permissions))
      os.makedirs(filepath, permissions)
   else:
      debug.log("Warning: The directory "+ filepath +" already exists")
   return filepath