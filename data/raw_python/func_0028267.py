def check_imc_creds(auth, url):
    """Function takes input of auth class object auth object and URL and returns a BOOL of TRUE
     if the authentication was successful.

    >>> auth = IMCAuth("http://", "10.101.0.203", "8080", "admin", "admin")

    >>> check_imc_creds(auth.creds, auth.url)
    True

    """
    test_url = '/imcrs'
    f_url = url + test_url
    try:
        response = requests.get(f_url, auth=auth, headers=HEADERS, verify=False)
        return bool(response.status_code == 200)
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " test_imc_creds: An Error has occured"