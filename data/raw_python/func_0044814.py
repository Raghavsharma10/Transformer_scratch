def rebase(self, text, char='X'):
        """
        Rebases text with stop words removed.
        """
        regexp = re.compile(r'\b(%s)\b' % '|'.join(self.collection),
                            re.IGNORECASE | re.UNICODE)

        def replace(m):
            word = m.group(1)
            return char * len(word)

        return regexp.sub(replace, text)