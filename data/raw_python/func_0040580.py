def strip_mentions_links(self, text):
        """ Strips Mentions and Links

            :param text: Text to be stripped from.
        """
        #print 'Before:', text
        new_text = [word for word in text.split() if not self.is_mention_line(word)]
        #print 'After:', u' '.join(new_text)
        return u' '.join(new_text)