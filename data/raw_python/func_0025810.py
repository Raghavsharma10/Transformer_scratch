def showStatus(self, msg, keep=0, cat=None):
        """ Show the given status string, but not until any given delay from
            the previous message has expired. keep is a time (secs) to force
            the message to remain without being overwritten or cleared. cat
            is a string category used only in the historical log. """
        # prep it, space-wise
        msg = msg.strip()
        if len(msg) > 0:
            # right here is the ideal place to collect a history of messages
            forhist = msg
            if cat: forhist = '['+cat+'] '+msg
            forhist = time.strftime("%a %H:%M:%S")+': '+forhist
            self._msgHistory.append(forhist)
            # now set the spacing
            msg = '  '+msg

        # stop here if it is a category not shown in the GUI
        if cat == DBG:
            return

        # see if we can show it
        now = time.time()
        if now >= self._leaveStatusMsgUntil: # we are clear, can show a msg
            # first see if this msg is '' - if so we will show an important
            # waiting msg instead of the '', and then pop it off our list
            if len(msg) < 1 and len(self._statusMsgsToShow) > 0:
                msg, keep = self._statusMsgsToShow[0] # overwrite both args
                del self._statusMsgsToShow[0]
            # now actuall print the status out to the status widget
            self.top.status.config(text = msg)
            # reset our delay flag
            self._leaveStatusMsgUntil = 0
            if keep > 0:
                self._leaveStatusMsgUntil = now + keep
        else:
            # there is a previous message still up, is this one important?
            if len(msg) > 0 and keep > 0:
                # Uh-oh, this is an important message that we don't want to
                # simply skip, but on the other hand we can't show it yet...
                # So we add it to _statusMsgsToShow and show it later (asap)
                if (msg,keep) not in self._statusMsgsToShow:
                    if len(self._statusMsgsToShow) < 7:
                        self._statusMsgsToShow.append( (msg,keep) ) # tuple
                        # kick off timer loop to get this one pushed through
                        if len(self._statusMsgsToShow) == 1:
                            self._pushMessages()
                    else:
                        # should never happen, but just in case
                        print("Lost message!: "+msg+" (too far behind...)")