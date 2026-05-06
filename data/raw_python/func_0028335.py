def set_imc_creds(h_url=None, imc_server=None, imc_port=None, imc_user=None,imc_pw=None):
    """ This function prompts user for IMC server information and credentuials and stores
    values in url and auth global variables"""
    global auth, url
    if h_url is None:
        imc_protocol = input(
        "What protocol would you like to use to connect to the IMC server: \n Press 1 for HTTP: \n Press 2 for HTTPS:")
        if imc_protocol == "1":
            h_url = 'http://'
        else:
            h_url = 'https://'
        imc_server = input("What is the ip address of the IMC server?")
        imc_port = input("What is the port number of the IMC server?")
        imc_user = input("What is the username of the IMC eAPI user?")
        imc_pw = input('''What is the password of the IMC eAPI user?''')
    url = h_url + imc_server + ":" + imc_port
    auth = requests.auth.HTTPDigestAuth(imc_user, imc_pw)
    test_url = '/imcrs'
    f_url = url + test_url
    try:
        r = requests.get(f_url, auth=auth, headers=headers, verify=False)
        print (r.status_code)
        return auth
    # checks for reqeusts exceptions
    except requests.exceptions.RequestException as e:
        print("Error:\n" + str(e))
        print("\n\nThe IMC server address is invalid. Please try again\n\n")
        set_imc_creds()
    if r.status_code != 200:  # checks for valid IMC credentials
        print("Error: \n You're credentials are invalid. Please try again\n\n")
        set_imc_creds()
    else:
        print("You've successfully access the IMC eAPI")