def __add_query_comment(sql):
        """
        Adds a comment line to the query to be executed containing the line number of the calling
        function.  This is useful for debugging slow queries, as the comment will show in the slow
        query log

        @type sql: str
        @param sql: sql needing comment
        @return:
        """
        # Inspect the call stack for the originating call
        file_name = ''
        line_number = ''
        caller_frames = inspect.getouterframes(inspect.currentframe())
        for frame in caller_frames:
            if "ShapewaysDb" not in frame[1]:
                file_name = frame[1]
                line_number = str(frame[2])
                break

        comment = "/*COYOTE: Q_SRC: {file}:{line} */\n".format(file=file_name, line=line_number)
        return comment + sql,