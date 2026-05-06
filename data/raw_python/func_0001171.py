def _proc_group(self, style, adopt=True):
        """Return whether group is "default" or "override".

        In the case of "override", the self.fields pre-format and post-format
        processors will be set under the "override" key.

        Parameters
        ----------
        style : dict
            A style that follows the schema defined in pyout.elements.
        adopt : bool, optional
            Merge `self.style` and `style`, giving priority to the latter's
            keys when there are conflicts.  If False, treat `style` as a
            standalone style.
        """
        fields = self.fields
        if style is not None:
            if adopt:
                style = elements.adopt(self.style, style)
            elements.validate(style)

            for column in self.columns:
                fields[column].add(
                    "pre", "override",
                    *(self.procgen.pre_from_style(style[column])))
                fields[column].add(
                    "post", "override",
                    *(self.procgen.post_from_style(style[column])))
            return "override"
        else:
            return "default"