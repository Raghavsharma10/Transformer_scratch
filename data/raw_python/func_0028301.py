def modify_telnet_template(auth, url, telnet_template, template_name= None, template_id = None):
    """
    Function takes input of a dictionry containing the required key/value pair for the modification
    of a telnet template.

    :param auth:
    :param url:
    :param telnet_template: Human readable label which is the name of the specific telnet template
    :param template_id Internal IMC number which designates the specific telnet template
    :return: int value of HTTP response code 201 for proper creation or 404 for failed creation
    :rtype int

    Sample of proper KV pairs. Please see documentation for valid values for different fields.

    telnet_template = {"type": "0",
    "name": "User_with_Enable",
    "authType": "4",
    "userName": "newadmin",
    "userPassword": "newpassword",
    "superPassword": "newpassword",
    "authTypeStr": "Username + Password + Super/Manager Password",
    "timeout": "4",
    "retries": "1",
    "port": "23",
    "version": "1",
    "creator": "admin",
    "accessType": "1",
    "operatorGroupStr": ""}
    """
    if template_name is None:
        template_name = telnet_template['name']
    if template_id is None:
        telnet_templates = get_telnet_template(auth, url)
        template_id = None
        for template in telnet_templates:
            if template['name'] == template_name:
                template_id = template['id']
    f_url = url + "/imcrs/plat/res/telnet/"+str(template_id)+"/update"
    response = requests.put(f_url, data = json.dumps(telnet_template), auth=auth, headers=HEADERS)
    try:
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " modify_telnet_template: An Error has occured"