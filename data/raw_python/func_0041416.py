def unarchive(filename,output_dir='.'):
    '''unpacks the given archive into ``output_dir``'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for archive in archive_formats:
        if filename.endswith(archive_formats[archive]['suffix']):
            return subprocess.call(archive_formats[archive]['command'](output_dir,filename))==0
    return False