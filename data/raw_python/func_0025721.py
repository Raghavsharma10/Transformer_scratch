def getAppDir():
    """ Return our application dir.  Create it if it doesn't exist. """
    # Be sure the resource dir exists
    theDir = os.path.expanduser('~/.')+APP_NAME.lower()
    if not os.path.exists(theDir):
        try:
            os.mkdir(theDir)
        except OSError:
            print('Could not create "'+theDir+'" to save GUI settings.')
            theDir = "./"+APP_NAME.lower()
    return theDir