def compose(self, *args, **kwargs):
        """
        Generate a file from the current template and given arguments.

        Warning:
            Make certain to check the formatted editor for correctness!

        Args:
            args: Positional arguments to update the template
            kwargs: Keyword arguments to update the template

        Returns:
            editor: An editor containing the formatted template.
        """
        linebreak = kwargs.pop("linebreak", "\n")
        # Update the internally stored args/kwargs from which formatting arguments come
        if len(args) > 0:
            self.args = args
        self._update(**kwargs)
        # Format string arguments (for the modified template)
        fkwargs = {}    # Format string keyword arguments
        modtmpl = []    # The modified template lines
        #curpos = 0      # Positional argument counter
        #i = 0
        for line in self:
            cline = copy(line)
            # If any special formatters exist, handle them
            for match in self._regex.findall(line):
                search = "[{}]".format("|".join(match))
                name, indent, delim, qual, _ = match
                if indent != "":
                    indent = " "*int(indent)
                delim = delim.replace("\\|", "|")
                # Collect and format the data accordingly
                data = getattr(self, name, None)
                # If no data exists, treat as optional
                if data is None:
                    cline = cline.replace(search, "")
                    continue
                elif delim.isdigit():
                    fkwargs[name] = getattr(self, "_fmt_"+name)()
                else:
                    fkwargs[name] = linebreak.join([indent+k+delim+qual+v+qual for k, v in data.items()])
                cline = cline.replace(search, "{"+name+"}")
            modtmpl.append(cline)
        modtmpl = "\n".join(modtmpl)
        print(modtmpl)
        dct = self.get_kwargs()
        dct.update(fkwargs)
        return self._constructor(textobj=modtmpl.format(*self.args, **dct))