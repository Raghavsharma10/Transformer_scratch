def find(self, query, threshold=None):
        """Simply return the best match to the query, None on no match.

        >>> from ngram import NGram
        >>> n = NGram(["Spam","Eggs","Ham"], key=lambda x:x.lower(), N=1)
        >>> n.find('Hom')
        'Ham'
        >>> n.find("Spom")
        'Spam'
        >>> n.find("Spom", 0.8)
        """
        results = self.search(query, threshold)
        if results:
            return results[0][0]
        else:
            return None