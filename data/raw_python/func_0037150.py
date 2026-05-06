def bump(self, kind=None, prerelease=None, inplace=True):
        '''
        Increment the version and/or pre-release value

        Parameters
        ----------

        kind : str
            Increment the version. Can be ``major``, ``minor``, or ``patch``,
            corresponding to the three segments of the version number (in
            order). A value of ``None`` will not increment the version number
            (default).

        prerelease : str
            Increment the version's pre-release value. Can be ``alpha`` or
            ``beta``. A prerelease value of ``None`` will remove a pre-release
            value if it exists (default).

        inplace : bool
            If false, returns a new ``BumpableVersion`` instance. If ``True``
            (default), bumps the version in place.


        Examples
        --------

        The ``kind`` argument increments the version:

        .. code-block:: python

            >>> v = BumpableVersion('1.0.1')
            >>> v.bump('patch')
            >>> v
            BumpableVersion ('1.0.2')
            >>>
            >>> v.bump('minor')
            >>> v
            BumpableVersion ('1.1')
            >>>
            >>> v.bump('minor')
            >>> v
            BumpableVersion ('1.2')
            >>>
            >>> v.bump('major')
            >>> v
            BumpableVersion ('2.0')
            >>>
            >>> v.bump('release')   # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ValueError: Bump kind "release" not understood

        The prerelease argument increments the pre-release value. If ``kind``
        is not supplied simultaneously the version is bumped with a patch
        before entering pre-release:

        .. code-block:: python

            >>> v = BumpableVersion('1.0.0')
            >>> v.bump(prerelease='alpha')
            >>> v
            BumpableVersion ('1.0.1a1')
            >>>
            >>> v.bump(prerelease='alpha')
            >>> v
            BumpableVersion ('1.0.1a2')
            >>>
            >>> v.bump(prerelease='beta')
            >>> v
            BumpableVersion ('1.0.1b1')
            >>>
            >>> v.bump('minor')
            >>> v
            BumpableVersion ('1.1')
            >>>
            >>> v.bump('minor', prerelease='beta')
            >>> v
            BumpableVersion ('1.2b1')
            >>>
            >>> v.bump(prerelease='beta')
            >>> v
            BumpableVersion ('1.2b2')
            >>>
            >>> v.bump('minor')
            >>> v
            BumpableVersion ('1.2')
            >>>
            >>> v.bump('minor', prerelease='beta')
            >>> v
            BumpableVersion ('1.3b1')
            >>>
            >>> v.bump('major', prerelease='alpha')
            >>> v
            BumpableVersion ('2.0a1')
            >>>
            >>> v.bump('major')
            >>> v
            BumpableVersion ('3.0')
            >>>
            >>> v.bump('patch', prerelease='beta')
            >>> v
            BumpableVersion ('3.0.1b1')
            >>>
            >>> v.bump('patch')
            >>> v
            BumpableVersion ('3.0.1')
            >>>
            >>> v.bump(prerelease='gamma')   # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ValueError: Prerelease type "gamma" not understood

        Releases cannot move from beta to alpha without a new major/minor/patch
        bump:

        .. code-block:: python

            >>> v = BumpableVersion('0.2b1')
            >>> v.bump(prerelease='alpha')    # doctest: \
            +ELLIPSIS +NORMALIZE_WHITESPACE
            Traceback (most recent call last):
            ...
            ValueError: Cannot bump version "0.2b1" to prerelease stage \
            "alpha" - version already in beta
            >>> v.bump('minor')
            >>> v
            BumpableVersion ('0.2')
            >>>


        Versions can return a new version or can be bumped in-place (default):

        .. code-block:: python

            >>> v = BumpableVersion('0.2')
            >>> v.bump('minor', inplace=False)
            BumpableVersion ('0.3')
            >>> v
            BumpableVersion ('0.2')

        '''

        if kind is not None:

            # if already in pre-release and we want to move to pre-release,
            # increment version + prerelease
            if self.prerelease and prerelease:
                new_prerelease = self._increment_prerelease(None, prerelease)
                new_version = self._increment_version(self.version, kind)

            # if already in pre-release and we want to exit pre-release,
            # remove prerelease
            elif self.prerelease:
                new_prerelease = None
                if self.version[2] == 0:
                    if kind == 'minor':
                        new_version = self.version
                    else:
                        new_version = self._increment_version(
                            self.version, kind)
                else:
                    if kind == 'patch':
                        new_version = self.version
                    else:
                        new_version = self._increment_version(
                            self.version, kind)

            else:
                new_prerelease = self._increment_prerelease(None, prerelease)
                new_version = self._increment_version(self.version, kind)

        elif prerelease is not None:
            if self.prerelease:
                new_prerelease = self._increment_prerelease(
                    self.prerelease, prerelease)
                new_version = self.version

            else:
                new_prerelease = self._increment_prerelease(None, prerelease)
                new_version = self._increment_version(self.version, 'patch')

        else:
            # default is bump patch

            new_prerelease = None
            new_version = self._increment_version(self.version, 'patch')

        if inplace:
            self.version = new_version
            self.prerelease = new_prerelease

        else:
            new = BumpableVersion()
            new.version = new_version
            new.prerelease = new_prerelease

            return new