def add(self, name, nestable, create_dir=True, update=False,
            label_func=str, template_subs=False):
        """
        Add a level to the nest

        :param string name: Name of the level. Forms the key in the output
            dictionary.
        :param nestable: Either an iterable object containing values, _or_ a
            function which takes a single argument (the control dictionary)
            and returns an iterable object containing values
        :param boolean create_dir: Should a directory level be created for this
            nestable?
        :param boolean update: Should the control dictionary be updated with
            the results of each value returned by the nestable? Only valid for
            dictionary results; useful for updating multiple values. At a
            minimum, a key-value pair corresponding to ``name`` must be
            returned.
        :param label_func: Function to be called to convert each value to a
            directory label.
        :param boolean template_subs: Should the strings in / returned by
            nestable be treated as templates? If true, str.format is called
            with the current values of the control dictionary.
        """
        # Convert everything to functions
        if not callable(nestable):
            if not _is_iter(nestable):
                raise ValueError("Invalid nestable: " + str(nestable))
            if is_string(nestable):
                warnings.warn(
                        "Passed a string as an iterable for name {0}".format(name))
            old_nestable = nestable
            nestable = _repeat_iter(old_nestable)
        if template_subs:
            nestable = _templated(nestable)

        new_controls = []
        for outdir, control in self._controls:
            for r in nestable(control):
                new_outdir, new_control = outdir, control.copy()
                if update:
                    # Make sure expected key exists
                    if name not in r:
                        raise KeyError("Missing key for {0}".format(name))
                    # Check for collisions
                    u = frozenset(control.keys()) & frozenset(r.keys())
                    if u:
                        msg = "Key overlap: {0}".format(u)
                        if self.fail_on_clash:
                            raise KeyError(msg)
                        elif self.warn_on_clash:
                            warnings.warn(msg)
                    new_control.update(r)
                    to_label = r[name]
                else:
                    new_control[name] = to_label = r

                if create_dir:
                    new_outdir = os.path.join(outdir, label_func(to_label))
                if self.include_outdir:
                    new_control['OUTDIR'] = new_outdir
                new_controls.append((new_outdir, new_control))

        self._controls = new_controls