def pic_loggedallowedremoterelieve_v1(self):
    """Update the receiver link sequence."""
    log = self.sequences.logs.fastaccess
    rec = self.sequences.receivers.fastaccess
    log.loggedallowedremoterelieve[0] = rec.r[0]