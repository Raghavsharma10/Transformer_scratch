def send_mail_worker(config, mail, event):
    """Worker task to send out an email, which is a blocking process unless it is threaded"""
    log = ""

    try:
        if config.get('ssl', True):
            server = SMTP_SSL(config['server'], port=config['port'], timeout=30)
        else:
            server = SMTP(config['server'], port=config['port'], timeout=30)

        if config['tls']:
            log += 'Starting TLS\n'
            server.starttls()

        if config['username'] != '':
            log += 'Logging in with ' + str(config['username']) + "\n"
            server.login(config['username'], config['password'])
        else:
            log += 'No username, trying anonymous access\n'

        log += 'Sending Mail\n'
        response_send = server.send_message(mail)
        server.quit()

    except timeout as e:
        log += 'Could not send email: ' + str(e) + "\n"
        return False, log, event

    log += 'Server response:' + str(response_send)
    return True, log, event