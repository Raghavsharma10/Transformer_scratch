def add_ip_scope(auth, url,startIp, endIp, name, description):
    """
    Function takes input of four strings Start Ip, endIp, name, and description to add new Ip Scope to terminal access
    in the HPE IMC base platform
    :param startIp: str Start of IP address scope ex. '10.101.0.1'
    :param endIp: str End of IP address scope ex. '10.101.0.254'
    :param name: str Name of the owner of this IP scope  ex. 'admin'
    :param description: str description of the Ip scope
    :return:
    """
    if auth is None or url is None:  # checks to see if the imc credentials are already available
        set_imc_creds()

    add_ip_scope_url = "/imcrs/res/access/assignedIpScope"
    f_url = url + add_ip_scope_url
    payload = ('''{  "startIp": "%s", "endIp": "%s","name": "%s","description": "%s" }'''
               %(str(startIp), str(endIp), str(name), str(description)))
    r = requests.post(f_url, auth=auth, headers=HEADERS, data=payload) # creates the URL using the payload variable as the contents
    try:
        if r.status_code == 200:
            print("IP Scope Successfully Created")
            return r.status_code
        elif r.status_code == 409:
            print ("IP Scope Already Exists")
            return r.status_code
    except requests.exceptions.RequestException as e:
            return "Error:\n" + str(e) + " add_ip_scope: An Error has occured"


    #Add host to IP scope
    #http://10.101.0.203:8080/imcrs/res/access/assignedIpScope/ip?ipScopeId=1
    '''{
      "ip": "10.101.0.1",
      "name": "Cisco2811.lab.local",
      "description": "Cisco 2811",
      "parentId": "1"
    }'''