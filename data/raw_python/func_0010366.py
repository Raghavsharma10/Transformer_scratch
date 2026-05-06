def download(url, localFileName=None, localDirName=None):
    """
    Utility function for downloading files from the web 
    and retaining the same filename.
    """
    localName = url2name(url)
    req = Request(url)
    r = urlopen(req)
    if r.info().has_key('Content-Disposition'):
        # If the response has Content-Disposition, we take file name from it
        localName = r.info()['Content-Disposition'].split('filename=')
        if len(localName) > 1:
            localName = localName[1]
            if localName[0] == '"' or localName[0] == "'":
                localName = localName[1:-1]
        else:
            localName = url2name(r.url)
    elif r.url != url:
        # if we were redirected, the real file name we take from the final URL
        localName = url2name(r.url)
    if localFileName:
        # we can force to save the file as specified name
        localName = localFileName
    if localDirName:
        # we can also put it in some custom directory
        if not os.path.exists(localDirName):
            os.makedirs(localDirName)
        localName = os.path.join(localDirName, localName)

    f = open(localName, 'wb')
    f.write(r.read())
    f.close()