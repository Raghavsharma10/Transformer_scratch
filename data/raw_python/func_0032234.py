def beforeRender(self, ctx):
        """
        Implement this hook to initialize the L{initialPerson} and
        L{initialState} slots with information from the request url's query
        args.
        """
        # see the comment in Organizer.urlForViewState which suggests an
        # alternate implementation of this kind of functionality.
        request = inevow.IRequest(ctx)
        if not set(['initial-person', 'initial-state']).issubset( # <=
            set(request.args)):
            return
        initialPersonName = request.args['initial-person'][0].decode('utf-8')
        initialPerson = self.store.findFirst(
            Person, Person.name == initialPersonName)
        if initialPerson is None:
            return
        initialState = request.args['initial-state'][0].decode('utf-8')
        if initialState not in ORGANIZER_VIEW_STATES.ALL_STATES:
            return
        self.initialPerson = initialPerson
        self.initialState = initialState