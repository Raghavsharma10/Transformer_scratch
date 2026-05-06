def find_by_ext(dirname, ext):
    """Find all files in a directory by extension."""
    # Get all fasta-files
    try: 
        files = os.listdir(dirname) 
    except OSError: 
        if os.path.exists(dirname): 
            cmd = "find {0} -maxdepth 1 -name \"*\"".format(dirname) 
            p = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE) 
            stdout, _stderr = p.communicate() 
            files = [os.path.basename(fname) for fname in stdout.decode().splitlines()] 
        else: 
            raise 
     
    retfiles = [os.path.join(dirname, fname) for fname in files if 
                    os.path.splitext(fname)[-1] in ext] 
 
    return retfiles