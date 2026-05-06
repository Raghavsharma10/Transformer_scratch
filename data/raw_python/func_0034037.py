def generic_send_mail(sender, dests, subject, message, key, origin='', html_message=False):
    """Generic mail sending function"""

    # If no EBUIO Mail settings have been set, then no e-mail shall be sent
    if settings.EBUIO_MAIL_SECRET_KEY and settings.EBUIO_MAIL_SECRET_HASH:
        headers = {}

        if key:
            from Crypto.Cipher import AES

            hash_key = hashlib.sha512(key + settings.EBUIO_MAIL_SECRET_HASH).hexdigest()[30:42]

            encrypter = AES.new(((settings.EBUIO_MAIL_SECRET_KEY) * 32)[:32], AES.MODE_CFB, '87447JEUPEBU4hR!')
            encrypted_key = encrypter.encrypt(hash_key + ':' + key)

            base64_key = base64.urlsafe_b64encode(encrypted_key)

            headers = {'Reply-To': settings.MAIL_SENDER.replace('@', '+' + base64_key + '@')}

        msg = EmailMessage(subject, message, sender, dests, headers=headers)
        if html_message:
            msg.content_subtype = "html"  # Main content is now text/html
        msg.send(fail_silently=False)

        try:
            from main.models import MailSend

            MailSend(dest=','.join(dests), subject=subject, sender=sender, message=message, origin=origin).save()

        except ImportError:
            pass
    else:
        logger.debug(
            "E-Mail notification not sent, since no EBUIO_MAIL_SECRET_KEY and EBUIO_MAIL_SECRET_HASH set in settingsLocal.py.")