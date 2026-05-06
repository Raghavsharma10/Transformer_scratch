def sniff_extension(file_path,verbose=True):
    '''sniff_extension will attempt to determine the file type based on the extension,
    and return the proper mimetype
    :param file_path: the full path to the file to sniff
    :param verbose: print stuff out
    '''
    mime_types =    { "xls": 'application/vnd.ms-excel',
                      "xlsx": 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                      "xml": 'text/xml',
                      "ods": 'application/vnd.oasis.opendocument.spreadsheet',
                      "csv": 'text/plain',
                      "tmpl": 'text/plain',
                      "pdf":  'application/pdf',
                      "php": 'application/x-httpd-php',
                      "jpg": 'image/jpeg',
                      "png": 'image/png',
                      "gif": 'image/gif',
                      "bmp": 'image/bmp',
                      "txt": 'text/plain',
                      "doc": 'application/msword',
                      "js": 'text/js',
                      "swf": 'application/x-shockwave-flash',
                      "mp3": 'audio/mpeg',
                      "zip": 'application/zip',
                      "simg": 'application/zip',
                      "rar": 'application/rar',
                      "tar": 'application/tar',
                      "arj": 'application/arj',
                      "cab": 'application/cab',
                      "html": 'text/html',
                      "htm": 'text/html',
                      "default": 'application/octet-stream',
                      "folder": 'application/vnd.google-apps.folder',
                      "img" : "application/octet-stream" }

    ext = os.path.basename(file_path).split('.')[-1]

    mime_type = mime_types.get(ext,None)

    if mime_type == None:
        mime_type = mime_types['txt']

    if verbose==True:
        bot.info("%s --> %s" %(file_path, mime_type))

    return mime_type