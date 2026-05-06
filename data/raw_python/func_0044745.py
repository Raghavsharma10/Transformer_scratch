def sendmail(array):
    """function for read data in db and send mail
    """
    username = 'root@robin8.io'
    FROM = username
    TO = [array["to"]]
    SUBJECT = array["subject"]
    # check on correct data in optional
    if type(array["optional"]) == str and array["optional"]:
        TEXT = array["optional"]
    else:
        return "Error: missed argument"
    # make template
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s""" % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP('localhost')
        logging.info(server)
        #server.ehlo()
        #server.starttls()
        # authorizing user, must setup your account
        #server.login(username, password)
        # send mail
        server.sendmail(FROM, TO, message)
        server.quit()
        logging.info(message)
        return "Success"
    except:
        return "Error"