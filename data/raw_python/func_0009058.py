def zip_up(file_list,zip_name,output_folder=None):
    '''zip_up will zip up some list of files into a package (.zip)
    :param file_list: a list of files to include in the zip.
    :param output_folder: the output folder to create the zip in. If not 
    :param zip_name: the name of the zipfile to return.
    specified, a temporary folder will be given.
    '''
    tmpdir = tempfile.mkdtemp()
   
    # Make a new archive    
    output_zip = "%s/%s" %(tmpdir,zip_name)
    zf = zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, allowZip64=True)

    # Write files to zip, depending on type
    for filename,content in file_list.items():

        bot.debug("Adding %s to package..." %filename)

        # If it's the files list, move files into the archive
        if filename.lower() == "files":
            if not isinstance(content,list): 
                content = [content]
            for copyfile in content:
                zf.write(copyfile,os.path.basename(copyfile))
                os.remove(copyfile)

        else:

            output_file = "%s/%s" %(tmpdir, filename)
        
            # If it's a list, write to new file, and save
            if isinstance(content,list):
                write_file(output_file,"\n".join(content))
        
            # If it's a dict, save to json
            elif isinstance(content,dict):
                write_json(content,output_file)

            # If bytes, need to decode
            elif isinstance(content,bytes):
                write_file(output_file,content.decode('utf-8'))
   
            # String or other
            else: 
                output_file = write_file(output_file,content)

            if os.path.exists(output_file):
                zf.write(output_file,filename)
                os.remove(output_file)

    # Close the zip file    
    zf.close()

    if output_folder is not None:
        shutil.copyfile(output_zip,"%s/%s"%(output_folder,zip_name))
        shutil.rmtree(tmpdir)
        output_zip = "%s/%s"%(output_folder,zip_name)

    return output_zip