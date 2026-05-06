def download(date_array, tag, sat_id, data_path, user=None, password=None):
    """
    Download SuperDARN data from Virginia Tech organized for loading by pysat.

    """

    import sys
    import os
    import pysftp
    import davitpy
    
    if user is None:
        user = os.environ['DBREADUSER']
    if password is None:
        password = os.environ['DBREADPASS']
        
    with pysftp.Connection(
            os.environ['VTDB'], 
            username=user,
            password=password) as sftp:
            
        for date in date_array:
            myDir = '/data/'+date.strftime("%Y")+'/grdex/'+tag+'/'
            fname = date.strftime("%Y%m%d")+'.' + tag + '.grdex'
            local_fname = fname+'.bz2'
            saved_fname = os.path.join(data_path,local_fname) 
            full_fname = os.path.join(data_path,fname) 
            try:
                print('Downloading file for '+date.strftime('%D'))
                sys.stdout.flush()
                sftp.get(myDir+local_fname, saved_fname)
                os.system('bunzip2 -c '+saved_fname+' > '+ full_fname)
                os.system('rm ' + saved_fname)
            except IOError:
                print('File not available for '+date.strftime('%D'))

    return