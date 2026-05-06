def import_tokens(self, tokens, import_hook=None, ignorecase=True):
        ''' Import a list of string as tokens '''
        text = self.text.lower() if ignorecase else self.text
        has_hooker = import_hook and callable(import_hook)
        cfrom = 0
        if self.__tokens:
            for tk in self.__tokens:
                if tk.cfrom and tk.cfrom > cfrom:
                    cfrom = tk.cfrom
        for token in tokens:
            if has_hooker:
                import_hook(token)
            to_find = token.lower() if ignorecase else token
            start = text.find(to_find, cfrom)
            # stanford parser
            if to_find == '``' or to_find == "''":
                start_dq = text.find('"', cfrom)
                if start_dq > -1 and (start == -1 or start > start_dq):
                    to_find = '"'
                    start = start_dq
            if to_find == '`' or to_find == "'":
                start_dq = text.find("'", cfrom)
                if start_dq > -1 and (start == -1 or start > start_dq):
                    to_find = "'"
                    start = start_dq
            if start == -1:
                raise LookupError('Cannot find token `{t}` in sent `{s}`({l}) from {i} ({p})'.format(t=token, s=self.text, l=len(self.text), i=cfrom, p=self.text[cfrom:cfrom + 20]))
            cfrom = start
            cto = cfrom + len(to_find)
            self.new_token(token, cfrom, cto)
            cfrom = cto - 1