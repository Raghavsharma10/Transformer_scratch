def frontiersZipInput(zip_path, output_prefix, download=None):
    """
    This method provides support for Frontiers production using base zipfiles
    as the input for ePub creation. It expects a valid pathname for one of the
    two zipfiles, and that both zipfiles are present in the same directory.
    """
    log.debug('frontiersZipInput called')
    #If there is a problem with the input, it should clearly describe the issue
    pathname, pathext = os.path.splitext(zip_path)
    path, name = os.path.split(pathname)
    if not pathext == '.zip':  # Checks for a path to zipfile
        log.error('Pathname provided does not end with .zip')
        print('Invalid file path: Does not have a zip extension.')
        sys.exit(1)
    #Construct the pair of zipfile pathnames
    file_root = name.split('-r')[0]
    zipname1 = "{0}-r{1}.zip".format(file_root, '1')
    zipname2 = "{0}-r{1}.zip".format(file_root, '2')
    #Construct the pathnames for output
    output = os.path.join(output_prefix, file_root)
    if os.path.isdir(output):
        shutil.rmtree(output)  # Delete previous output
    output_meta = os.path.join(output, 'META-INF')
    images_output = os.path.join(output, 'EPUB', 'images')
    with zipfile.ZipFile(os.path.join(path, zipname1), 'r') as xml_zip:
        zip_dir = '{0}-r1'.format(file_root)
        xml = '/'.join([zip_dir, '{0}.xml'.format(file_root)])
        try:
            xml_zip.extract(xml)
        except KeyError:
            log.critical('There is no item {0} in the zipfile'.format(xml))
            sys.exit('There is no item {0} in the zipfile'.format(xml))
        else:
            if not os.path.isdir(output_meta):
                os.makedirs(output_meta)
            shutil.copy(xml, os.path.join(output_meta))
            os.remove(xml)
            os.rmdir(zip_dir)
    with zipfile.ZipFile(os.path.join(path, zipname2), 'r') as image_zip:
        zip_dir = '{0}-r2'.format(file_root)
        for i in image_zip.namelist():
            if 'image_m' in i:
                image_zip.extract(i)
        if not os.path.isdir(images_output):
            os.makedirs(images_output)
        unzipped_images = os.path.join(zip_dir, 'images', 'image_m')
        for i in os.listdir(unzipped_images):
            shutil.copy(os.path.join(unzipped_images, i), images_output)
        shutil.rmtree(zip_dir)
    return file_root