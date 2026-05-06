def refresh(self, count_common=4, min_common=1000, timeout=20):
        """
        Generate a new sentence
        :param int count_common: the number of words with minimal commonness
        :param int min_common: the minimal commonness based on Google common word list
        :param float timeout: time in seconds to timeout
        :return list of str: return tokens on success

        >>> GeneratePassword().refresh()
        ['The', 'men', 'in', 'power', 'are', 'committed', 'in', 'principle', 'to', 'modernization', ',', 'but', 'economic', 'and', 'social', 'changes', 'are', 'proceeding', 'only', 'erratically', '.']
        """
        start = time()
        while time() - start < timeout:
            tokens = [token for token, pos in self.brown.get_tagged_sent()]
            current_count = 0
            for word, commonness in self.sentence_tool.rate(tokens):
                if commonness > min_common:
                    current_count += 1
                if current_count >= count_common:
                    self.tokens = tokens
                    return self.tokens

        raise TimeoutError