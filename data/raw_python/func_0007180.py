def download_file_powershell(url, target):
    '''
    Download the file at url to target using Powershell (which will validate
    trust). Raise an exception if the command cannot complete.
    '''
    target = os.path.abspath(target)
    cmd = [
        'powershell',
        '-Command',
        '(new-object System.Net.WebClient).DownloadFile(%(url)r, %(target)r)' % vars(),
    ]
    subprocess.check_call(cmd)