def add_attachment(message, attachment, rfc2231=True):
    '''Attach an attachment to a message as a side effect.

    Arguments:
        message: MIMEMultipart instance.
        attachment: Attachment instance.
    '''
    data = attachment.read()

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(data)
    encoders.encode_base64(part)
    filename = attachment.name if rfc2231 else Header(attachment.name).encode()
    part.add_header('Content-Disposition', 'attachment',
                    filename=filename)

    message.attach(part)