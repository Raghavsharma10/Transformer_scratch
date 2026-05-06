def _like_pattern(start, end, anywhere, index, length):
        """
        Create a LIKE pattern to use as a search parameter for a WHERE clause.

        :param start: Value to be found at the start
        :param end: Value to be found at the end
        :param anywhere: Value to be found anywhere
        :param index: Value to be found at a certain index
        :param length: Minimum character length
        :return: WHERE pattern
        """
        # Unpack index tuple
        index_num, index_char = index
        index = None

        # Start, end, anywhere
        if all(i for i in [start, end, anywhere]) and not any(i for i in [index, length]):
            return '{start}%{anywhere}%{end}'.format(start=start, end=end, anywhere=anywhere)

        # Start, end
        elif all(i for i in [start, end]) and not any(i for i in [anywhere, index, length]):
            return '{start}%{end}'.format(start=start, end=end)

        # Start, anywhere
        elif all(i for i in [start, anywhere]) and not any(i for i in [end, index, length]):
            return '{start}%{anywhere}%'.format(start=start, anywhere=anywhere)

        # End, anywhere
        elif all(i for i in [end, anywhere]) and not any(i for i in [start, index, length]):
            return '%{anywhere}%{end}'.format(end=end, anywhere=anywhere)

        # Start
        elif start and not any(i for i in [end, anywhere, index, length]):
            return '{start}%'.format(start=start)

        # End
        elif end and not any(i for i in [start, anywhere, index, length]):
            return '%{end}'.format(end=end)

        # Anywhere
        elif anywhere and not any(i for i in [start, end, index, length]):
            return '%{anywhere}%'.format(anywhere=anywhere)

        # Index
        elif index_num and index_char and not any(i for i in [start, end, anywhere, length]):
            return '{index_num}{index_char}%'.format(index_num='_' * (index_num + 1), index_char=index_char)

        # Length
        elif length and not any(i for i in [start, end, anywhere, index]):
            return '{length}'.format(length='_%' * length)

        else:
            return None