def get_revision(self, location):
        """
        Return the maximum revision for all files under a given location
        """
        # Note: taken from setuptools.command.egg_info
        revision = 0

        for base, dirs, files in os.walk(location):
            if self.dirname not in dirs:
                dirs[:] = []
                continue    # no sense walking uncontrolled subdirs
            dirs.remove(self.dirname)
            entries_fn = os.path.join(base, self.dirname, 'entries')
            if not os.path.exists(entries_fn):
                ## FIXME: should we warn?
                continue
            f = open(entries_fn)
            data = f.read()
            f.close()

            if data.startswith('8') or data.startswith('9') or data.startswith('10'):
                data = list(map(str.splitlines, data.split('\n\x0c\n')))
                del data[0][0]  # get rid of the '8'
                dirurl = data[0][3]
                revs = [int(d[9]) for d in data if len(d)>9 and d[9]]+[0]
                if revs:
                    localrev = max(revs)
                else:
                    localrev = 0
            elif data.startswith('<?xml'):
                dirurl = _svn_xml_url_re.search(data).group(1)    # get repository URL
                revs = [int(m.group(1)) for m in _svn_rev_re.finditer(data)]+[0]
                if revs:
                    localrev = max(revs)
                else:
                    localrev = 0
            else:
                logger.warn("Unrecognized .svn/entries format; skipping %s", base)
                dirs[:] = []
                continue
            if base == location:
                base_url = dirurl+'/'   # save the root url
            elif not dirurl.startswith(base_url):
                dirs[:] = []
                continue    # not part of the same svn tree, skip it
            revision = max(revision, localrev)
        return revision