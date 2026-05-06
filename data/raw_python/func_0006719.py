def starts_with(self, prefix):
        """
        Find all words starting with a prefix.

        Args:
            prefix: A prefix to be searched for.

        Returns:
            A list of all words found.
        """
        prefix = prefix.lower()
        found_words = []

        res = cgaddag.gdg_starts_with(self.gdg, prefix.encode(encoding="ascii"))
        tmp = res

        while tmp:
            word = tmp.contents.str.decode("ascii")
            found_words.append(word)
            tmp = tmp.contents.next

        cgaddag.gdg_destroy_result(res)
        return found_words