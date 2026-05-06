def get_snmp_templates(auth, url, template_name=None):
    """
    Takes no input, or template_name as input to issue RESTUL call to HP IMC

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param template_name: str value of template name

    :return list object containing one or more dictionaries where each dictionary represents one
    snmp template

    :rtype list

    """
    f_url = url + "/imcrs/plat/res/snmp?start=0&size=10000&desc=false&total=false"
    response = requests.get(f_url, auth=auth, headers=HEADERS)
    try:
        if response.status_code == 200:
            snmp_templates = (json.loads(response.text))
            template = None
            if type(snmp_templates['snmpParamTemplate']) is dict:
                my_templates = [snmp_templates['snmpParamTemplate']]
                snmp_templates['snmpParamTemplate'] = my_templates
            if template_name is None:
                return snmp_templates['snmpParamTemplate']
            elif template_name is not None:
                for snmp_template in snmp_templates['snmpParamTemplate']:
                    if snmp_template['name'] == template_name:
                        template = [snmp_template]
                if template == None:
                    return 404
                else:
                    return template
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " get_snmp_templates: An Error has occured"