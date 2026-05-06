def getunzipped(username, repo, thedir):
    """Downloads and unzips a zip file"""
    theurl = "https://github.com/" + username + "/" + repo + "/archive/master.zip"
    name = os.path.join(thedir, 'temp.zip')
    try:
        name = urllib.urlretrieve(theurl, name)
        name = os.path.join(thedir, 'temp.zip')
    except IOError as e:
        print("Can't retrieve %r to %r: %s" % (theurl, thedir, e))
        return
    try:
        z = zipfile.ZipFile(name)
    except zipfile.error as e:
        print("Bad zipfile (from %r): %s" % (theurl, e))
        return
    z.extractall(thedir)
    z.close()
    os.remove(name)

    copy_tree(os.path.join(thedir, repo + "-master"), thedir)
    shutil.rmtree(os.path.join(thedir, repo + "-master"))