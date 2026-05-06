def create_ssh_template(auth, url, ssh_template ):
    """
    Function takes input of a dictionry containing the required key/value pair for the creation
    of a ssh template.

    :param auth:
    :param url:
    :param ssh: dictionary of valid JSON which complains to API schema
    :return: int value of HTTP response code 201 for proper creation or 404 for failed creation
    :rtype int

    Sample of proper KV pairs. Please see documentation for valid values for different fields.

    ssh_template = {
    "type": "0",
    "name": "ssh_admin_template",
    "authType": "3",
    "authTypeStr": "Password + Super Password",
    "userName": "admin",
    "password": "password",
    "superPassword": "password",
    "port": "22",
    "timeout": "10",
    "retries": "3",
    "keyFileName": "",
    "keyPhrase": ""
    }
    """
    f_url = url + "/imcrs/plat/res/ssh/add"
    response = requests.post(f_url, data = json.dumps(ssh_template), auth=auth, headers=HEADERS)
    try:
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " create_ssh_template: An Error has occured"