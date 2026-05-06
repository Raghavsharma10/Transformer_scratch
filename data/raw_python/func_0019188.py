def pic_loggedrequiredremoterelease_v2(self):
    """Update the receiver link sequence."""
    log = self.sequences.logs.fastaccess
    rec = self.sequences.receivers.fastaccess
    log.loggedrequiredremoterelease[0] = rec.s[0]