def epub_zip(outdirect):
    """
    Zips up the input file directory into an EPUB file.
    """

    def recursive_zip(zipf, directory, folder=None):
        if folder is None:
            folder = ''
        for item in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, item)):
                zipf.write(os.path.join(directory, item),
                           os.path.join(directory, item))
            elif os.path.isdir(os.path.join(directory, item)):
                recursive_zip(zipf, os.path.join(directory, item),
                              os.path.join(folder, item))

    log.info('Zipping up the directory {0}'.format(outdirect))
    epub_filename = outdirect + '.epub'
    epub = zipfile.ZipFile(epub_filename, 'w')
    current_dir = os.getcwd()
    os.chdir(outdirect)
    epub.write('mimetype')
    log.info('Recursively zipping META-INF and EPUB')
    for item in os.listdir('.'):
        if item == 'mimetype':
            continue
        recursive_zip(epub, item)
    os.chdir(current_dir)
    epub.close()