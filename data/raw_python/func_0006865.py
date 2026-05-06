def file_unzipper(directory):
   """ This function will unzip all files in the runroot directory and
   subdirectories
   """
   debug.log("Unzipping directory (%s)..."%directory)
   #FINDING AND UNZIPPING ZIPPED FILES
   for root, dirs, files in os.walk(directory, topdown=False):
      if root != "":
         orig_dir = os.getcwd()
         os.chdir(directory)
         Popen('gunzip -q -f *.gz > /dev/null 2>&1', shell=True).wait()
         Popen('unzip -qq -o "*.zip" > /dev/null 2>&1', shell=True).wait()
         Popen('rm -f *.zip > /dev/null 2>&1', shell=True).wait()
         os.chdir(orig_dir)