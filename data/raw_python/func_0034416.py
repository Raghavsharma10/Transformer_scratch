def send_zipfile(request, fileList):
    """                                                                         
    Create a ZIP file on disk and transmit it in chunks of 8KB,                 
    without loading the whole file into memory. A similar approach can          
    be used for large dynamic PDF files.                                        
    """
    temp = tempfile.TemporaryFile()
    archive = zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED)
    for artist,files in fileList.iteritems():
        for f in files:
            archive.write(f[0], '%s/%s' % (artist, f[1]))
    archive.close()
    wrapper = FixedFileWrapper(temp)
    response = HttpResponse(wrapper, content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename=FrogSources.zip'
    response['Content-Length'] = temp.tell()
    temp.seek(0)
    return response