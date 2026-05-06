def split(cls, text):
        """split(text : string) -> [string]

        Splits 'text' into multiple paragraphs and return a list of each
        paragraph.
        """
        result = [line.strip('\n') for line in cls.parasep_re.split(text)]
        if result == ['', '']:
            result = ['']
        return result