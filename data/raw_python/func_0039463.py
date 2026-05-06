def _metahash(self):
        """Checksum hash of all the inputs to this rule.

        Output is invalid until collect_srcs and collect_deps have been run.

        In theory, if this hash doesn't change, the outputs won't change
        either, which makes it useful for caching.
        """

        # BE CAREFUL when overriding/extending this method. You want to copy
        # the if(cached)/return(cached) part, then call this method, then at
        # the end update the cached metahash. Just like this code, basically,
        # only you call the method from the base class in the middle of it. If
        # you get this wrong it could result in butcher not noticing changed
        # inputs between runs, which could cause really nasty problems.
        # TODO(ben): the above warning seems avoidable with better memoization

        if self._cached_metahash:
            return self._cached_metahash

        # If you are extending this function in a subclass,
        # here is where you do:
        # BaseBuilder._metahash(self)

        log.debug('[%s]: Metahash input: %s', self.address,
                  unicode(self.address))
        mhash = util.hash_str(unicode(self.address))
        log.debug('[%s]: Metahash input: %s', self.address, self.rule.params)
        mhash = util.hash_str(str(self.rule.params), hasher=mhash)
        for src in self.rule.source_files or []:
            log.debug('[%s]: Metahash input: %s', self.address, src)
            mhash = util.hash_str(src, hasher=mhash)
            mhash = util.hash_file(self.srcs_map[src], hasher=mhash)
        for dep in self.rule.composed_deps() or []:
            dep_rule = self.rule.subgraph.node[dep]['target_obj']
            for item in dep_rule.output_files:
                log.debug('[%s]: Metahash input: %s', self.address, item)
                item_path = os.path.join(self.buildroot, item)
                mhash = util.hash_str(item, hasher=mhash)
                mhash = util.hash_file(item_path, hasher=mhash)
        self._cached_metahash = mhash
        return mhash