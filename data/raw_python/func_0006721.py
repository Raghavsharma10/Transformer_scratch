def ends_with(self, suffix):
        """
        Find all words ending with a suffix.

        Args:
            suffix: A suffix to be searched for.

        Returns:
            A list of all words found.
        """
        suffix = suffix.lower()
        found_words = []

        res = cgaddag.gdg_ends_with(self.gdg, suffix.encode(encoding="ascii"))
        tmp = res

        while tmp:
            word = tmp.contents.str.decode("ascii")
            found_words.append(word)
            tmp = tmp.contents.next

        cgaddag.gdg_destroy_result(res)
        return found_words