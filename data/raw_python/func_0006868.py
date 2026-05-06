def copy_dir(src, dst):
   """ this function will simply copy the file from the source path to the dest
   path given as input
   """
   try:
      debug.log("copy dir from "+ src, "to "+ dst)
      shutil.copytree(src, dst)
   except Exception as e:
      debug.log("Error: happened while copying!\n%s\n"%e)