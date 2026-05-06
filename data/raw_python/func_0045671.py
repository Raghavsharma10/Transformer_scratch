def find(self, *strings, **kwargs):
        """
        Search the entire editor for lines that match the string.

        .. code-block:: Python

            string = '''word one
            word two
            three'''
            ed = Editor(string)
            ed.find('word')          # [(0, "word one"), (1, "word two")]
            ed.find('word', 'three') # {'word': [...], 'three': [(2, "three")]}

        Args:
            strings (str): Any number of strings to search for
            keys_only (bool): Only return keys
            start (int): Optional line to start searching on
            stop (int): Optional line to stop searching on

        Returns:
            results: If multiple strings searched a dictionary of string key, (line number, line) values (else just values)
        """
        start = kwargs.pop("start", 0)
        stop = kwargs.pop("stop", None)
        keys_only = kwargs.pop("keys_only", False)
        results = {string: [] for string in strings}
        stop = len(self) if stop is None else stop
        for i, line in enumerate(self[start:stop]):
            for string in strings:
                if string in line:
                    if keys_only:
                        results[string].append(i)
                    else:
                        results[string].append((i, line))
        if len(strings) == 1:
            return results[strings[0]]
        return results