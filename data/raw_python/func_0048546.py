def from_verb(cls, verb):
        """
        Constructs a :class:`Spoolverb` instance from the string
        representation of the given verb.

        Args:
            verb (str): representation of the verb e.g.:
                ``'ASCRIBESPOOL01LOAN12/150526150528'``. Can also be in
                binary format (:obj:`bytes`): ``b'ASCRIBESPOOL01PIECE'``.

        Returns:
            :class:`Spoolverb` instance.

        """
        pattern = r'^(?P<meta>[A-Z]+)(?P<version>\d+)(?P<action>[A-Z]+)(?P<arg1>\d+)?(\/(?P<arg2>\d+))?$'
        try:
            verb = verb.decode()
        except AttributeError:
            pass
        match = re.match(pattern, verb)
        if not match:
            raise SpoolverbError('Invalid spoolverb: {}'.format(verb))

        data = match.groupdict()
        meta = data['meta']
        version = data['version']
        action = data['action']
        if action == 'EDITIONS':
            num_editions = data['arg1']
            return cls(meta=meta, version=version, action=action, num_editions=int(num_editions))
        elif action == 'LOAN':
            # TODO Review. Workaround for piece loans
            try:
                edition_num = int(data['arg1'])
            except TypeError:
                edition_num = 0
            loan_start = data['arg2'][:6]
            loan_end = data['arg2'][6:]
            return cls(meta=meta, version=version, action=action, edition_num=int(edition_num),
                       loan_start=loan_start, loan_end=loan_end)
        elif action in ['FUEL', 'PIECE', 'CONSIGNEDREGISTRATION']:
            # no edition number for these verbs
            return cls(meta=meta, version=version, action=action)
        else:
            edition_num = data['arg1']
            return cls(meta=meta, version=version, action=action, edition_num=int(edition_num))