def get_image_tar(image_path):
    '''get an image tar, either written in memory or to
    the file system. file_obj will either be the file object,
    or the file itself.
    '''
    bot.debug('Generate file system tar...')   
    file_obj = Client.image.export(image_path=image_path)
    if file_obj is None:
        bot.error("Error generating tar, exiting.")
        sys.exit(1)
    tar = tarfile.open(file_obj)
    return file_obj, tar