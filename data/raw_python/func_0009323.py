def _contains_wildcards(cls, s):
        """
        Return True if the string contains any unquoted special characters
        (question-mark or asterisk), otherwise False.

        Ex: _contains_wildcards("foo") => FALSE
        Ex: _contains_wildcards("foo\?") => FALSE
        Ex: _contains_wildcards("foo?") => TRUE
        Ex: _contains_wildcards("\*bar") => FALSE
        Ex: _contains_wildcards("*bar") => TRUE

        :param string s: string to check
        :returns: True if string contains any unquoted special characters,
            False otherwise.
        :rtype: boolean

        This function is a support function for _compare().
        """

        idx = s.find("*")
        if idx != -1:
            if idx == 0:
                return True
            else:
                if s[idx - 1] != "\\":
                    return True

        idx = s.find("?")
        if idx != -1:
            if idx == 0:
                return True
            else:
                if s[idx - 1] != "\\":
                    return True
        return False