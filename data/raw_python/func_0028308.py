def modify_snmp_template(auth, url, snmp_template, template_name= None, template_id = None):
    """
    Function takes input of a dictionry containing the required key/value pair for the modification
    of a snmp template.

    :param auth:
    :param url:
    :param ssh_template: Human readable label which is the name of the specific ssh template
    :param template_id Internal IMC number which designates the specific ssh template
    :return: int value of HTTP response code 201 for proper creation or 404 for failed creation
    :rtype int

    Sample of proper KV pairs. Please see documentation for valid values for different fields.

    snmp_template =  {
      "version": "2",
      "name": "new_snmp_template",
      "type": "0",
      "paraType": "SNMPv2c",
      "roCommunity": "newpublic",
      "rwCommunity": "newprivate",
      "timeout": "4",
      "retries": "4",
      "contextName": "",
      "securityName": " ",
      "securityMode": "1",
      "authScheme": "0",
      "authPassword": "",
      "privScheme": "0",
      "privPassword": "",
      "snmpPort": "161",
      "isAutodiscoverTemp": "1",
      "creator": "admin",
      "accessType": "1",
      "operatorGroupStr": ""
      }
    """
    if template_name is None:
        template_name = snmp_template['name']
    if template_id is None:
        snmp_templates = get_snmp_templates(auth, url)
        template_id = None
        for template in snmp_templates:
            if template['name'] == template_name:
                template_id = template['id']
    f_url = url + "/imcrs/plat/res/snmp/"+str(template_id)+"/update"
    response = requests.put(f_url, data = json.dumps(snmp_template), auth=auth, headers=HEADERS)
    try:
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " modify_snmp_template: An Error has occured"