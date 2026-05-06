def check_rev_options(self, rev, dest, rev_options):
        """Check the revision options before checkout to compensate that tags
        and branches may need origin/ as a prefix.
        Returns the SHA1 of the branch or tag if found.
        """
        revisions = self.get_tag_revs(dest)
        revisions.update(self.get_branch_revs(dest))

        origin_rev = 'origin/%s' % rev
        if origin_rev in revisions:
            # remote branch
            return [revisions[origin_rev]]
        elif rev in revisions:
            # a local tag or branch name
            return [revisions[rev]]
        else:
            logger.warn("Could not find a tag or branch '%s', assuming commit." % rev)
            return rev_options