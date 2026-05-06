def contains(self, sub):
        """
        Find all words containing a substring.

        Args:
            sub: A substring to be searched for.

        Returns:
            A list of all words found.
        """
        sub = sub.lower()
        found_words = set()

        res = cgaddag.gdg_contains(self.gdg, sub.encode(encoding="ascii"))
        tmp = res

        while tmp:
            word = tmp.contents.str.decode("ascii")
            found_words.add(word)
            tmp = tmp.contents.next

        cgaddag.gdg_destroy_result(res)
        return list(found_words)