def pic_loggedrequiredremoterelease_v1(self):
    """Update the receiver link sequence."""
    log = self.sequences.logs.fastaccess
    rec = self.sequences.receivers.fastaccess
    log.loggedrequiredremoterelease[0] = rec.d[0]