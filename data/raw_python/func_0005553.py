def send_mail(form, from_name):
    """
    Sends a mail using the configured mail server for Pulsar.  See mailgun documentation at
    https://documentation.mailgun.com/en/latest/user_manual.html#sending-via-api for specifics.
 
    Args:
        form: `dict`. The mail form fields, i.e. 'to', 'from', ...

    Returns: 
        `requests.models.Response` instance.

    Raises: 
        `requests.exceptions.HTTPError`: The status code is not ok.
        `Exception`: The environment variable MAILGUN_DOMAIN or MAILGUN_API_KEY isn't set. 

    Example::

        payload = {
            "from"="{} <mailgun@{}>".format(from_name, pulsarpy.MAIL_DOMAIN), 
            "subject": "mailgun test", 
            "text": "howdy there",
            "to": "nathankw@stanford.edu", 
        }
        send_mail(payload)

    """
    form["from"] = "{} <mailgun@{}>".format(from_name, pulsarpy.MAIL_DOMAIN),
    if not pulsarpy.MAIL_SERVER_URL:
        raise Exception("MAILGUN_DOMAIN environment variable not set.")
    if not pulsarpy.MAIL_AUTH[1]:
        raise Exception("MAILGUN_API_KEY environment varible not set.")
    res = requests.post(pulsarpy.MAIL_SERVER_URL, data=form, auth=pulsarpy.MAIL_AUTH)
    res.raise_for_status()
    return res