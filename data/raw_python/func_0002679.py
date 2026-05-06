def make_epub_base(location):
    """
    Creates the base structure for an EPUB file in a specified location.

    This function creates constant components for the structure of the EPUB in
    a specified directory location.

    Parameters
    ----------
    location : str
        A path string to a local directory in which the EPUB is to be built
    """
    log.info('Making EPUB base files in {0}'.format(location))
    with open(os.path.join(location, 'mimetype'), 'w') as out:  # mimetype file
        out.write('application/epub+zip')

    #Create EPUB and META-INF directorys
    os.mkdir(os.path.join(location, 'META-INF'))
    os.mkdir(os.path.join(location, 'EPUB'))
    os.mkdir(os.path.join(location, 'EPUB', 'css'))

    with open(os.path.join(location, 'META-INF', 'container.xml'), 'w') as out:
        out.write('''\
<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
   <rootfiles>
      <rootfile full-path="EPUB/package.opf" media-type="application/oebps-package+xml"/>
   </rootfiles>
</container>''')

    with open(os.path.join(location, 'EPUB', 'css', 'default.css') ,'wb') as out:
        out.write(bytes(DEFAULT_CSS, 'UTF-8'))