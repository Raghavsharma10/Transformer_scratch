def modify_ssh_template(auth, url, ssh_template, template_name= None, template_id = None):
    """
    Function takes input of a dictionry containing the required key/value pair for the modification
    of a ssh template.

    :param auth:
    :param url:
    :param ssh_template: Human readable label which is the name of the specific ssh template
    :param template_id Internal IMC number which designates the specific ssh template
    :return: int value of HTTP response code 201 for proper creation or 404 for failed creation
    :rtype int

    Sample of proper KV pairs. Please see documentation for valid values for different fields.

    ssh_template = {
    "type": "0",
    "name": "ssh_admin_template",
    "authType": "3",
    "authTypeStr": "Password + Super Password",
    "userName": "newadmin",
    "password": "password",
    "superPassword": "password",
    "port": "22",
    "timeout": "10",
    "retries": "3",
    "keyFileName": "",
    "keyPhrase": ""
    }
    """
    if template_name is None:
        template_name = ssh_template['name']
    if template_id is None:
        ssh_templates = get_ssh_template(auth, url)
        template_id = None
        for template in ssh_templates:
            if template['name'] == template_name:
                template_id = template['id']
    f_url = url + "/imcrs/plat/res/ssh/"+str(template_id)+"/update"
    response = requests.put(f_url, data = json.dumps(ssh_template), auth=auth, headers=HEADERS)
    try:
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " modify_ssh_template: An Error has occured"