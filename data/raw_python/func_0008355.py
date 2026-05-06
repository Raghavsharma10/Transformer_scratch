def _get_options_dic(self, options: List[str]) -> Dict[str, str]:
        """
        Convert the option list to a dictionary where the key is the option and the value is the related option.
        Is called in the init.

        :param options: options given to the plugin.
        :type options: List[str]
        :return: dictionary which contains the option key as str related to the option string
        :rtype Dict[str, str]
        """
        options_dic = {}
        for option in options:
            cur_option = option.split("=")
            if len(cur_option) != 2:
                self.log.warning(f"'{option}' is not valid and will be ignored.")
            options_dic[cur_option[0]] = cur_option[1]
        return options_dic