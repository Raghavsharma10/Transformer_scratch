def __parse_target(targetstr, current_repo=None):
        """Parse a build target string.

        General form: //repo[gitref]/dir/path:target.

        These are all valid:
          //repo
          //repo[a038fi31d9e8bc11582ef1b1b1982d8fc]
          //repo[a039aa30853298]:foo
          //repo/dir
          //repo[a037928734]/dir
          //repo/dir/path
          //repo/dir/path:foo
          :foo
          dir/path
          dir/path:foo
          dir:foo

        Returns: {'repo': '//reponame',
                  'git_ref': 'a839a38fd...',
                  'path': 'dir/path',
                  'target': 'targetname}
        """
        # 'blah' -> ':blah'
        if not (':' in targetstr or '/' in targetstr):
            targetstr = ':%s' % targetstr

        match = re.match(
            r'^(?://(?P<repo>[\w-]+)(?:\[(?P<git_ref>.*)\])?)?'
            r'(?:$|/?(?P<path>[\w/-]+)?(?::?(?P<target>[\w-]+)?))', targetstr)
        try:
            groups = match.groupdict()
            if not groups['repo']:
                groups['repo'] = current_repo
            if not groups['git_ref']:
                groups['git_ref'] = 'develop'
            if not groups['target']:
                groups['target'] = 'all'
            if not groups['path']:
                groups['path'] = ''
        except AttributeError:
            raise error.ButcherError('"%s" is not a valid build target.')
        #log.debug('parse_target: %s -> %s', targetstr, groups)
        return groups