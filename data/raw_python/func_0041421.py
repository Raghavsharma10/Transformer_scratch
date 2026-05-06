def find(file):
    '''tries to find ``file`` using OS-specific searches and some guessing'''
    # Try MacOS Spotlight:
    mdfind = which('mdfind')
    if mdfind:
        out = run([mdfind,'-name',file],stderr=None,quiet=None)
        if out.return_code==0 and out.output:
                for fname in out.output.split('\n'):
                    if os.path.basename(fname)==file:
                        return fname

    # Try UNIX locate:
    locate = which('locate')
    if locate:
        out = run([locate,file],stderr=None,quiet=None)
        if out.return_code==0 and out.output:
            for fname in out.output.split('\n'):
                if os.path.basename(fname)==file:
                    return fname

    # Try to look through the PATH, and some guesses:
    path_search = os.environ["PATH"].split(os.pathsep)
    path_search += ['/usr/local/afni','/usr/local/afni/atlases','/usr/local/share','/usr/local/share/afni','/usr/local/share/afni/atlases']
    afni_path = which('afni')
    if afni_path:
        path_search.append(os.path.dirname(afni_path))
    if nl.wrappers.fsl.bet2:
        path_search.append(os.path.dirname(nl.wrappers.fsl.bet2))
    for path in path_search:
        path = path.strip('"')
        try:
            if file in os.listdir(path):
                return os.path.join(path,file)
        except:
            pass