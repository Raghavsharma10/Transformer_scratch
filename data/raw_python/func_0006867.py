def copy_file(src, dst, ignore=None):
   """ this function will simply copy the file from the source path to the dest
   path given as input
   """
   # Sanity checkpoint
   src = re.sub('[^\w/\-\.\*]', '', src)
   dst = re.sub('[^\w/\-\.\*]', '', dst)
   if len(re.sub('[\W]', '', src)) < 5 or len(re.sub('[\W]', '', dst)) < 5:
      debug.log("Error: Copying file failed. Provided paths are invalid! src='%s' dst='%s'"%(src, dst))
   else:
      # Check destination
      check = False
      if dst[-1] == '/':
         if os.path.exists(dst):
            check = True # Valid Dir
         else:
            debug.log("Error: Copying file failed. Destination directory does not exist (%s)"%(dst)) #DEBUG
      elif os.path.exists(dst):
         if os.path.isdir(dst):
            check = True # Valid Dir
            dst += '/' # Add missing slash
         else:
            debug.log("Error: Copying file failed. %s exists!"%dst)
      elif os.path.exists(os.path.dirname(dst)):
         check = True # Valid file path
      else:
         debug.log("Error: Copying file failed. %s is an invalid distination!"%dst)
      if check:
         # Check source
         files = glob.glob(src)
         if ignore is not None: files = [fil for fil in files if not ignore in fil]
         if len(files) != 0:
            debug.log("Copying File(s)...", "Copy from %s"%src, "to %s"%dst) #DEBUG
            for file_ in files:
               # Check file exists
               if os.path.isfile(file_):
                  debug.log("Copying file: %s"%file_) #DEBUG
                  shutil.copy(file_, dst)
               else:
                  debug.log("Error: Copying file failed. %s is not a regular file!"%file_) #DEBUG
         else: debug.log("Error: Copying file failed. No files were found! (%s)"%src)