def _reset(self):
        '''
            _reset - reset this object. Assigned to .reset after __init__ call.
        '''
        HTMLParser.reset(self)

        self.root = None
        self.doctype = None
        self._inTag = []