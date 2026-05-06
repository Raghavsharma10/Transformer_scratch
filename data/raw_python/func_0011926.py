def hyphenify(self, ascii=False):
        """Turn non-word characters (incl. underscore) into single hyphens.
        If ascii=True, return ASCII-only.
        If also lossless=True, use the UTF-8 codes for the non-ASCII characters.
        """
        s = str(self)
        s = re.sub("""['"\u2018\u2019\u201c\u201d]""", '', s)  # quotes
        s = re.sub(r'(?:\s|%20)+', '-', s)  # whitespace
        if ascii == True:  # ASCII-only
            s = s.encode('ascii', 'xmlcharrefreplace').decode('ascii')  # use entities
        s = re.sub("&?([^;]*?);", r'.\1-', s)  # entities
        s = s.replace('#', 'u')
        s = re.sub(r"\W+", '-', s).strip(' -')
        return String(s)