def mentions(self):
        """
        Returns mentions

        :return: list of mentions
        :rtype: list

        """
        if self._mentions is None:
            self._mentions = []
            for mention_element in self._element.xpath('mention'):
                this_mention = Mention(self, mention_element)
                self._mentions.append(this_mention)
                if this_mention.representative:
                    self._representative = this_mention
        return self._mentions