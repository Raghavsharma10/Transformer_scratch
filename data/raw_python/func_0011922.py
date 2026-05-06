def camelify(self):
        """turn a string to CamelCase, omitting non-word characters"""
        outstring = self.titleify(allwords=True)
        outstring = re.sub(r"&[^;]+;", " ", outstring)
        outstring = re.sub(r"\W+", "", outstring)
        return String(outstring)