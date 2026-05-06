def generate_control_field(self, revision=None):
        """
        Generate a Debian control file field referring for this repository and revision.

        :param revision: A reference to a revision, most likely the name of a
                         branch (a string, defaults to :attr:`default_revision`).
        :returns: A tuple with two strings: The name of the field and the value.

        This generates a `Vcs-Bzr` field for Bazaar repositories, a `Vcs-Git`
        field for Git repositories and a `Vcs-Hg` field for Mercurial
        repositories. Here's an example based on the public git repository of
        the `vcs-repo-mgr` project:

        >>> from vcs_repo_mgr import coerce_repository
        >>> repository = coerce_repository('https://github.com/xolox/python-vcs-repo-mgr.git')
        >>> repository.generate_control_field()
        ('Vcs-Git', 'https://github.com/xolox/python-vcs-repo-mgr.git#b617731b6c0ca746665f597d2f24b8814b137ebc')
        """
        value = "%s#%s" % (self.remote or self.local, self.find_revision_id(revision))
        return self.control_field, value