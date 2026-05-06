def clone(url, tmpdir=None):
    '''clone a repository from Github'''
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    name = os.path.basename(url).replace('.git', '')
    dest = '%s/%s' %(tmpdir,name)
    return_code = os.system('git clone %s %s' %(url,dest))
    if return_code == 0:
        return dest
    bot.error('Error cloning repo.')
    sys.exit(return_code)