def _ass_needs_refresh(self, cached_ass, file_ass):
        """Checks if assistant needs refresh.

        Assistant needs refresh iff any of following conditions is True:
        - stored source file is different than given source file
        - stored assistant ctime is lower than current source file ctime
        - stored list of subassistants is different than given list of subassistants
        - stored ctime of any of the snippets that this assistant uses to compose
          args is lower than current ctime of that snippet

        Args:
            cached_ass: an assistant from cache hierarchy
                        (for format see Cache class docstring)
            file_ass: the respective assistant from filesystem hierarchy
                      (for format see what refresh_role accepts)
        """
        if cached_ass['source'] != file_ass['source']:
            return True
        if os.path.getctime(file_ass['source']) > cached_ass.get('ctime', 0.0):
            return True
        if set(cached_ass['subhierarchy'].keys()) != set(set(file_ass['subhierarchy'].keys())):
            return True
        for snip_name, snip_ctime in cached_ass['snippets'].items():
            if self._get_snippet_ctime(snip_name) > snip_ctime:
                return True

        return False