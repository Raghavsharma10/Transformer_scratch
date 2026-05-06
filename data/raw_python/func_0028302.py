def delete_telnet_template(auth, url, template_name= None, template_id= None):
    """
    Takes template_name as input to issue RESTUL call to HP IMC which will delete the specific
    telnet template from the IMC system

    :param auth: requests auth object #usually auth.creds from auth pyhpeimc.auth.class

    :param url: base url of IMC RS interface #usually auth.url from pyhpeimc.auth.authclass

    :param template_name: str value of template name

    :param template_id: str value template template_id value

    :return: int HTTP response code

    :rtype  int
    """
    try:
        if template_id is None:
            telnet_templates = get_telnet_template(auth, url)
            if template_name is None:
                template_name = telnet_template['name']
            template_id = None
            for template in telnet_templates:
                if template['name'] == template_name:
                    template_id = template['id']
        f_url = url + "/imcrs/plat/res/telnet/%s/delete" % template_id
        response = requests.delete(f_url, auth=auth, headers=HEADERS)
        return response.status_code
    except requests.exceptions.RequestException as error:
        return "Error:\n" + str(error) + " delete_telnet_template: An Error has occured"