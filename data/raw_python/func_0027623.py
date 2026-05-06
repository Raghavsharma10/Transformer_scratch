def get_help_msg(self,
                     dotspace_ending=False,  # type: bool
                     **kwargs):
        # type: (...) -> str
        """
        The method used to get the formatted help message according to kwargs. By default it returns the 'help_msg'
        attribute, whether it is defined at the instance level or at the class level.

        The help message is formatted according to help_msg.format(**kwargs), and may be terminated with a dot
        and a space if dotspace_ending is set to True.

        :param dotspace_ending: True will append a dot and a space at the end of the message if it is not
        empty (default is False)
        :param kwargs: keyword arguments to format the help message
        :return: the formatted help message
        """
        context = self.get_context_for_help_msgs(kwargs)

        if self.help_msg is not None and len(self.help_msg) > 0:
            # create a copy because we will modify it
            context = copy(context)

            # first format if needed
            try:
                help_msg = self.help_msg
                variables = re.findall("{\S+}", help_msg)
                for v in set(variables):
                    v = v[1:-1]
                    if v in context and len(str(context[v])) > self.__max_str_length_displayed__:
                        new_name = '@@@@' + v + '@@@@'
                        help_msg = help_msg.replace('{' + v + '}', '{' + new_name + '}')
                        context[new_name] = "(too big for display)"

                help_msg = help_msg.format(**context)

            except KeyError as e:
                # no need to raise from e, __cause__ is set in the constructor
                raise HelpMsgFormattingException(self.help_msg, e, context)

            # then add a trailing dot and space if needed
            if dotspace_ending:
                return end_with_dot_space(help_msg)
            else:
                return help_msg
        else:
            return ''