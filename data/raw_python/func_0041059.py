def _parse(self, stream):
        """Parse a JSON BUILD file.

        Args:
          builddata: dictionary of buildfile data
          reponame: name of the repo that it came from
          path: directory path within the repo
        """
        builddata = json.load(stream)
        log.debug('This is a JSON build file.')

        if 'targets' not in builddata:
            log.warn('Warning: No targets defined here.')
            return

        for tdata in builddata['targets']:
            # TODO: validate name
            target = address.new(target=tdata.pop('name'),
                                 repo=self.target.repo,
                                 path=self.target.path)
            # Duplicate target definition? Uh oh.
            if target in self.node and 'target_obj' in self.node[target]:
                raise error.ButcherError(
                    'Target is defined more than once: %s', target)

            rule_obj = targets.new(name=target,
                                   ruletype=tdata.pop('type'),
                                   **tdata)

            log.debug('New target: %s', target)
            self.add_node(target, {'target_obj': rule_obj})

            # dep could be ":blabla" or "//foo:blabla" or "//foo/bar:blabla"
            for dep in rule_obj.composed_deps() or []:
                d_target = address.new(dep)
                if not d_target.repo:  # ":blabla"
                    d_target.repo = self.target.repo
                if d_target.repo == self.target.repo and not d_target.path:
                    d_target.path = self.target.path
                if d_target not in self.nodes():
                    self.add_node(d_target)
                log.debug('New dep: %s -> %s', target, d_target)
                self.add_edge(target, d_target)