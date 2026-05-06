def delete_image_tar(file_obj, tar):
    '''delete image tar will close a file object (if extracted into
    memory) or delete from the file system (if saved to disk)'''
    try:
        file_obj.close()
    except:
        tar.close()
    if os.path.exists(file_obj):
        os.remove(file_obj)
        deleted = True
        bot.debug('Deleted temporary tar.')   
    return deleted