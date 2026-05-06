def upload_file(storage_service,bucket,bucket_path,file_name,verbose=True):
    '''get_folder will return the folder with folder_name, and if create=True,
    will create it if not found. If folder is found or created, the metadata is
    returned, otherwise None is returned
    :param storage_service: the drive_service created from get_storage_service
    :param bucket: the bucket object from get_bucket
    :param file_name: the name of the file to upload
    :param bucket_path: the path to upload to
    '''
    # Set up path on bucket
    upload_path = "%s/%s" %(bucket['id'],bucket_path)
    if upload_path[-1] != '/':
        upload_path = "%s/" %(upload_path)
    upload_path = "%s%s" %(upload_path,os.path.basename(file_name))
    body = {'name': upload_path }
    # Create media object with correct mimetype
    if os.path.exists(file_name):
        mimetype = sniff_extension(file_name,verbose=verbose)
        media = http.MediaFileUpload(file_name,
                                     mimetype=mimetype,
                                     resumable=True)
        request = storage_service.objects().insert(bucket=bucket['id'], 
                                                   body=body,
                                                   predefinedAcl="publicRead",
                                                   media_body=media)
        result = request.execute()
        return result
    bot.warning('%s requested for upload does not exist, skipping' %file_name)