def send(self, send_to, from_who, subject, message, reply_to=None):
        """Send Email.

        To use this module pass in a message, send_to, from_who, and subject.

        :param send_to: ``str``
        :param from_who: ``str``
        :param subject: ``str``
        :param message: ``str``
        :param reply_to: ``str``
        """
        # Set the reply to address if it's None
        if reply_to is None:
            reply_to = from_who

        try:
            em_msg = text.MIMEText(
                _text=message.encode('utf8'),
                _subtype='plain',
                _charset='utf8'
            )
            em_msg["Subject"] = subject
            em_msg["From"] = from_who
            em_msg["To"] = send_to
            em_msg["Reply-To"] = reply_to

            # Send Customer Messages
            built_message = em_msg.as_string()

            self.smtp.sendmail(
                from_addr=em_msg["From"],
                to_addrs=em_msg["To"],
                msg=built_message
            )
        except Exception as exp:
            msg = 'Failed to send message due to "%s"' % exp
            self.log.error(msg)
            raise cloudlib.MessageFailure(msg)
        else:
            self.log.debug(message)
        finally:
            self.smtp.quit()