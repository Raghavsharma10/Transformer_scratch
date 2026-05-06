def get_ssh_template(auth, url, template_name=None):
    """
    Takes no input, or template_name as input to issue RESTUL call to HP IMC

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param template_name: str value of template name

    :return list object containing one or more dictionaries where each dictionary represents one
    ssh template

    :rtype list

    """
    f_url = url + "/imcrs/plat/res/ssh?start=0&size=10000&desc=false&total=false"
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            ssh_templates = (json.loads(response.text))
            template = None
            if type(ssh_templates['sshParamTemplate']) is dict:
                my_templates = [ssh_templates['sshParamTemplate']]
                ssh_templates['sshParamTemplate'] = my_templates
            if template_name is None:
                return ssh_templates['sshParamTemplate']
            elif template_name is not None:
                for ssh_template in ssh_templates['sshParamTemplate']:
                    if ssh_template['name'] == template_name:
                        template = [ssh_template]
                print (type(template))
                if template == None:
                    return 404
                else:
                    return template
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_ssh_templates: An Error has occured"