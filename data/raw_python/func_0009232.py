def search_string(self):
        """"
        Returns the JSON string that LendingClub expects for it's search
        """
        self.__normalize()

        # Get the template
        tmpl_source = unicode(open(self.tmpl_file).read())

        # Process template
        compiler = Compiler()
        template = compiler.compile(tmpl_source)
        out = template(self)
        if not out:
            return False
        out = ''.join(out)

        #
        # Cleanup output and remove all extra space
        #

        # remove extra spaces
        out = re.sub('\n', '', out)
        out = re.sub('\s{3,}', ' ', out)

        # Remove hanging commas i.e: [1, 2,]
        out = re.sub(',\s*([}\\]])', '\\1', out)

        # Space between brackets i.e: ],  [
        out = re.sub('([{\\[}\\]])(,?)\s*([{\\[}\\]])', '\\1\\2\\3', out)

        # Cleanup spaces around [, {, }, ], : and , characters
        out = re.sub('\s*([{\\[\\]}:,])\s*', '\\1', out)

        return out