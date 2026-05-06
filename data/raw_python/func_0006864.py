def file_zipper(root_dir):
   """ This function will zip the files created in the runroot directory and
   subdirectories """
   # FINDING AND ZIPPING UNZIPPED FILES
   for root, dirs, files in os.walk(root_dir, topdown=False):
      if root != "":
         if root[-1] != '/': root += '/'
         for current_file in files:
            filepath = "%s/%s"%(root, current_file)
            try:
               file_size = os.path.getsize(filepath)
            except Exception as e:
               file_size = 0
               debug.log('Error: file_zipper failed to zip following file '+filepath, e)
            # Excluding small files, gzipped files and links
            if (         file_size > 50
                 and     current_file[-3:] != ".gz"
                 and not os.path.islink(filepath)
               ):
               if current_file[-4:] == ".zip":
                  # Unzip file
                  ec = Popen('unzip -qq "%s" -d %s > /dev/null 2>&1'%(filepath, root), shell=True).wait()
                  if ec > 0:
                     debug.log('Error: fileZipper failed to unzip following file %s'%filepath)
                     continue
                  else:
                     ec = Popen('rm -f "%s" > /dev/null 2>&1'%(filepath), shell=True).wait()
                     if ec > 0: debug.log('Error: fileZipper failed to delete the original zip file (%s)'%filepath)
                     filepath = filepath[:-4]
                  # Saving a gzipped version
                  with open_(filepath, 'rb') as f, open_(filepath+".gz", 'wb', 9) as gz:
                     gz.writelines(f)
                  # Deleting old (non-zipped) file
                  try: os.remove(filepath)
                  except OSError as e:
                     debug.log(("WARNING! The file %s could not be "
                                    "removed!\n%s")%(current_file, e))