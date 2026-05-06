def launchBrowser(url, brow_bin='mozilla', subj=None):
    """ Given a URL, try to pop it up in a browser on most platforms.
    brow_bin is only used on OS's where there is no "open" or "start" cmd.
    """

    if not subj: subj = url

    # Tries to use webbrowser module on most OSes, unless a system command
    # is needed.  (E.g. win, linux, sun, etc)
    if sys.platform not in ('os2warp, iphone'): # try webbrowser w/ everything?
        import webbrowser
        if not webbrowser.open(url):
            print("Error opening URL: "+url)
        else:
            print('Help on "'+subj+'" is now being displayed in a web browser')
        return

    # Go ahead and fork a subprocess to call the correct binary
    pid = os.fork()
    if pid == 0: # child
        if sys.platform == 'darwin':
            if 0 != os.system('open "'+url+'"'): # does not seem to keep '#.*'
                print("Error opening URL: "+url)
        os._exit(0)
#       The following retries if "-remote" doesnt work, opening a new browser
#       cmd = brow_bin+" -remote 'openURL("+url+")' '"+url+"' 1> /dev/null 2>&1"
#       if 0 != os.system(cmd)
#           print "Running "+brow_bin+" for HTML help..."
#           os.execvp(brow_bin,[brow_bin,url])
#       os._exit(0)

    else: # parent
        if not subj:
            subj = url
        print('Help on "'+subj+'" is now being displayed in a browser')