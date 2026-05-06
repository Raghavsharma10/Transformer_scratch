def get_memory_tar(image_path):
    '''get an in memory tar of an image. Use carefully, not as reliable
       as get_image_tar
    '''
    byte_array = Client.image.export(image_path)
    file_object = io.BytesIO(byte_array)
    tar = tarfile.open(mode="r|*", fileobj=file_object)
    return (file_object,tar)