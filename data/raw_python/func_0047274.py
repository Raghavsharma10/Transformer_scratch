def get_title(self):
        """
        Get title line from .rst file.

        **中文文档**

        从一个 ``_filename`` 所指定的 .rst 文件中, 找到顶级标题.
        也就是第一个 ``====`` 或 ``----`` 或 ``~~~~`` 上面一行.
        """
        header_bar_char_list = "=-~+*#^"

        cursor_previous_line = None
        for cursor_line in textfile.readlines(self.rst_path, strip="both"):
            for header_bar_char in header_bar_char_list:
                if cursor_line.startswith(header_bar_char):
                    flag_full_bar_char = cursor_line == header_bar_char * len(cursor_line)
                    flag_line_length_greather_than_1 = len(cursor_line) >= 1
                    flag_previous_line_not_empty = bool(cursor_previous_line)
                    if flag_full_bar_char \
                            and flag_line_length_greather_than_1 \
                            and flag_previous_line_not_empty:
                        return cursor_previous_line.strip()
            cursor_previous_line = cursor_line

        msg = "Warning, this document doesn't have any %s header!" % header_bar_char_list
        return None