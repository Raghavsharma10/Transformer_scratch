def get_telnet_template(auth, url, template_name=None):
    """
    Takes no input, or template_name as input to issue RESTUL call to HP IMC

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param template_name: str value of template name

    :return list object containing one or more dictionaries where each dictionary represents one
    telnet template

    :rtype list

    """
    f_url = url + "/imcrs/plat/res/telnet?start=0&size=10000&desc=false&total=false"
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            telnet_templates = (json.loads(response.text))
            template = None
            if type(telnet_templates['telnetParamTemplate']) is dict:
                my_templates = [telnet_templates['telnetParamTemplate']]
                telnet_templates['telnetParamTemplate'] = my_templates
            if template_name is None:
                return telnet_templates['telnetParamTemplate']
            elif template_name is not None:
                for telnet_template in telnet_templates['telnetParamTemplate']:

                    if telnet_template['name'] == template_name:
                        template = [telnet_template]
                print (type(template))
                if template == None:
                    return 404
                else:
                    return template
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_telnet_templates: An Error has occured"