def download(date_array, tag, sat_id, data_path=None, user=None, password=None):
    """Downloads data from Madrigal.
    
    The user's names should be provided in field user. John Malkovich should be 
    entered as John+Malkovich 
    
    The password field should be the user's email address. These parameters
    are passed to Madrigal when downloading.
    
    The affiliation field is set to pysat to enable tracking of pysat downloads.
    
    Parameters
    ----------
    
    
    """
    import subprocess
    
    # currently passes things along if no user and password supplied
    # need to do this for testing
    # TODO, implement user and password values in test code
    # specific to DMSP
    if user is None:
        print ('No user information supplied for download.')
        user = 'pysat_testing'
    if password is None:
        print ('Please provide email address in password field.')
        password = 'pysat_testing@not_real_email.org'

    a = subprocess.check_output(["globalDownload.py", "--verbose", 
                    "--url=http://cedar.openmadrigal.org",
                    '--outputDir='+data_path,
                    '--user_fullname='+user,
                    '--user_email='+password,
                    '--user_affiliation=pysat',
                    '--format=hdf5',
                    '--startDate='+date_array[0].strftime('%m/%d/%Y'),
                    '--endDate='+date_array[-1].strftime('%m/%d/%Y'),
                    '--inst=8100',
                    '--kindat='+str(madrigal_tag[sat_id])])
    print ('Feedback from openMadrigal ', a)