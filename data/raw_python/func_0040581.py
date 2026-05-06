def normalize(self, text):
        """ Normalizes text.
            Converts to lowercase,
            Unicode NFC normalization
            and removes mentions and links

            :param text: Text to be normalized.
        """
        #print 'Normalize...\n'
        text = text.lower()
        text = unicodedata.normalize('NFC', text)
        text = self.strip_mentions_links(text)
        return text