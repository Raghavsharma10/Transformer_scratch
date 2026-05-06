def program_name(self):
        r"""The name of the script, callable from the command line.
        """
        name = "-".join(
            word.lower() for word in uqbar.strings.delimit_words(type(self).__name__)
        )
        return name