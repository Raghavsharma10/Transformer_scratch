def send_mail(subject, body, smtp_server, user, password, from_addr, to_addrs):
    ''' Sends a run status mail with the traceback to a specified E-Mail address if a run crashes.
    '''
    logging.info('Send status E-Mail (' + subject + ')')
    content = string.join((
        "From: %s" % from_addr,
        "To: %s" % ','.join(to_addrs),  # comma separated according to RFC822
        "Subject: %s" % subject,
        "",
        body),
        "\r\n")
    server = smtplib.SMTP_SSL(smtp_server)
    server.login(user, password)
    server.sendmail(from_addr, to_addrs, content)
    server.quit()