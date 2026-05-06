def expand_cmd_labels(self):
        """Expand make-style variables in cmd parameters.

        Currently:
        $(location <foo>)     Location of one dependency or output file.
        $(locations <foo>)    Space-delimited list of foo's output files.
        $(SRCS)               Space-delimited list of this rule's source files.
        $(OUTS)               Space-delimited list of this rule's output files.
        $(@D)                 Full path to the output directory for this rule.
        $@                    Path to the output (single) file for this rule.
        """
        cmd = self.cmd

        def _expand_onesrc():
            """Expand $@ or $(@) to one output file."""
            outs = self.rule.params['outs'] or []
            if len(outs) != 1:
                raise error.TargetBuildFailed(
                    self.address,
                    '$@ substitution requires exactly one output file, but '
                    'this rule has %s of them: %s' % (len(outs), outs))
            else:
                return os.path.join(self.buildroot, self.path_to_this_rule,
                                    outs[0])

        # TODO: this function is dumb and way too long
        def _expand_makevar(re_match):
            """Expands one substitution symbol."""
            # Expand $(location foo) and $(locations foo):
            label = None
            tagstr = re_match.groups()[0]
            tag_location = re.match(
                r'\s*location\s+([A-Za-z0-9/\-_:\.]+)\s*', tagstr)
            tag_locations = re.match(
                r'\s*locations\s+([A-Za-z0-9/\-_:\.]+)\s*', tagstr)
            if tag_location:
                label = tag_location.groups()[0]
            elif tag_locations:
                label = tag_locations.groups()[0]
            if label:
                # Is it a filename found in the outputs of this rule?
                if label in self.rule.params['outs']:
                    return os.path.join(self.buildroot, self.address.repo,
                                        self.address.path, label)
                # Is it an address found in the deps of this rule?
                addr = self.rule.makeaddress(label)
                if addr not in self.rule.composed_deps():
                    raise error.TargetBuildFailed(
                        self.address,
                        '%s is referenced in cmd but is neither an output '
                        'file from this rule nor a dependency of this rule.' %
                        label)
                else:
                    paths = [x for x in self.rulefor(addr).output_files]
                    if len(paths) is 0:
                        raise error.TargetBuildFailed(
                            self.address,
                            'cmd refers to %s, but it has no output files.')
                    elif len(paths) > 1 and tag_location:
                        raise error.TargetBuildFailed(
                            self.address,
                            'Bad substitution in cmd: Expected exactly one '
                            'file, but %s expands to %s files.' % (
                                addr, len(paths)))
                    else:
                        return ' '.join(
                            [os.path.join(self.buildroot, x) for x in paths])

            # Expand $(OUTS):
            elif re.match(r'OUTS', tagstr):
                return ' '.join(
                    [os.path.join(self.buildroot, x)
                     for x in self.rule.output_files])

            # Expand $(SRCS):
            elif re.match(r'SRCS', tagstr):
                return ' '.join(os.path.join(self.path_to_this_rule, x)
                                for x in self.rule.params['srcs'] or [])

            # Expand $(@D):
            elif re.match(r'\s*@D\s*', tagstr):
                ruledir = os.path.join(self.buildroot, self.path_to_this_rule)
                return ruledir

            # Expand $(@), $@:
            elif re.match(r'\s*@\s*', tagstr):
                return _expand_onesrc()

            else:
                raise error.TargetBuildFailed(
                    self.address,
                    '[%s] Unrecognized substitution in cmd: %s' % (
                        self.address, re_match.group()))

        cmd, _ = re.subn(self.paren_tag_re, _expand_makevar, cmd)

        # Match tags starting with $ without parens. Will also catch parens, so
        # this goes after the tag_re substitutions.
        cmd, _ = re.subn(self.noparen_tag_re, _expand_makevar, cmd)

        # Now that we're done looking for $(blabla) and $bla parameters, clean
        # up any $$ escaping:
        cmd, _ = re.subn(r'\$\$', '$', cmd)

        # Maybe try heuristic label expansion?  Actually on second thought
        # that's a terrible idea. Use the explicit syntax, you lazy slobs. ;-)

        # TODO: Maybe consider other expansions from the gnu make manual?
        # $^ might be useful.
        # http://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html#Automatic-Variables
        self.cmd = cmd