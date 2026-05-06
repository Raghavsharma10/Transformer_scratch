def create_zip_dir(zipfile_path, *file_list):
   """ This function creates a zipfile located in zipFilePath with the files in
   the file list
   # fileList can be both a comma separated list or an array
   """
   try:
      if isinstance(file_list, (list, tuple)): #unfolding list of list or tuple
         if len(file_list) == 1:
            if isinstance(file_list[0], (list, tuple)): file_list = file_list[0]
      #converting string to iterable list
      if isinstance(file_list, str): file_list = [file_list]
      if file_list:
         with ZipFile(zipfile_path, 'w') as zf:
            for cur_file in file_list:
               if '/' in cur_file:
                  os.chdir('/'.join(cur_file.split('/')[:-1]))
               elif '/' in zipfile_path:
                  os.chdir('/'.join(zipfile_path.split('/')[:-1]))
               zf.write(cur_file.split('/')[-1])
      else:
         debug.log('Error: No Files in list!',zipfile_path+' was not created!')
   except Exception as e:
      debug.log('Error: Could not create zip dir! argtype: '+
                 str(type(file_list)), "FileList: "+ str(file_list),
                 "Errormessage: "+ str(e))