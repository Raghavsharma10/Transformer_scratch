def move_file(src, dst):
   """ this function will simply move the file from the source path to the dest
   path given as input
   """
   # Sanity checkpoint
   src = re.sub('[^\w/\-\.\*]', '', src)
   dst = re.sub('[^\w/\-\.\*]', '', dst)
   if len(re.sub('[\W]', '', src)) < 5 or len(re.sub('[\W]', '', dst)) < 5:
      debug.log("Error: Moving file failed. Provided paths are invalid! src='%s' dst='%s'"%(src, dst))
   else:
      # Check destination
      check = False
      if dst[-1] == '/':
         if os.path.exists(dst):
            check = True # Valid Dir
         else:
            debug.log("Error: Moving file failed. Destination directory does not exist (%s)"%(dst)) #DEBUG
      elif os.path.exists(dst):
         if os.path.isdir(dst):
            check = True # Valid Dir
            dst += '/' # Add missing slash
         else:
            debug.log("Error: Moving file failed. %s exists!"%dst)
      elif os.path.exists(os.path.dirname(dst)):
         check = True # Valid file path
      else:
         debug.log("Error: Moving file failed. %s is an invalid distination!"%dst)
      if check:
         # Check source
         files = glob.glob(src)
         if len(files) != 0:
            debug.log("Moving File(s)...", "Move from %s"%src, "to %s"%dst)
            for file_ in files:
               # Check if file contains invalid symbols:
               invalid_chars = re.findall('[^\w/\-\.\*]', os.path.basename(file_))
               if invalid_chars:
                  debug.graceful_exit(("Error: File %s contains invalid "
                                      "characters %s!"
                                      )%(os.path.basename(file_), invalid_chars))
                  continue
               # Check file exists
               if os.path.isfile(file_):
                  debug.log("Moving file: %s"%file_)
                  shutil.move(file_, dst)
               else:
                  debug.log("Error: Moving file failed. %s is not a regular file!"%file_)
         else: debug.log("Error: Moving file failed. No files were found! (%s)"%src)