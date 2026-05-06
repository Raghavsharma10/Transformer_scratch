def clean_text(self, text):
        '''Clean text using bleach.'''
        if text is None:
            return ''
        text = re.sub(ILLEGAL_CHARACTERS_RE, '', text)
        if '<' in text or '&lt' in text:
            text = clean(text, tags=self.tags, strip=self.strip)

        return unescape(text)