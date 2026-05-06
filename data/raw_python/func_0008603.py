def open_file(path):
    """Opens Explorer/Finder with given path, depending on platform"""
    if sys.platform=='win32':
        os.startfile(path)
        #subprocess.Popen(['start', path], shell= True)
    
    elif sys.platform=='darwin':
        subprocess.Popen(['open', path])
    
    else:
        try:
            subprocess.Popen(['xdg-open', path])
        except OSError:
            pass