def send_mail_worker(config, mail, event):
    """Worker task to send out an email, which blocks the process unless it is threaded"""
    log = ""

    try:
        if config.mail_ssl:
            server = SMTP_SSL(config.mail_server, port=config.mail_server_port, timeout=30)
        else:
            server = SMTP(config.mail_server, port=config.mail_server_port, timeout=30)

        if config.mail_tls:
            log += 'Starting TLS\n'
            server.starttls()

        if config.mail_username != '':
            log += 'Logging in with ' + str(config.mail_username) + "\n"
            server.login(config.mail_username, config.mail_password)
        else:
            log += 'No username, trying anonymous access\n'

        log += 'Sending Mail\n'
        response_send = server.send_message(mail)
        server.quit()

    except timeout as e:
        log += 'Could not send email to enrollee, mailserver timeout: ' + str(e) + "\n"
        return False, log, event

    log += 'Server response:' + str(response_send)
    return True, log, event