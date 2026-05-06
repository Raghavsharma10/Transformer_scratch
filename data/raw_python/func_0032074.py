def createPortForm(self, req, tag):
        """
        Create and return a L{LiveForm} for adding a new L{TCPPort} or
        L{SSLPort} to the site store.
        """
        def port(s):
            n = int(s)
            if n < 0 or n > 65535:
                raise ValueError(s)
            return n

        factories = []
        for f in self.store.parent.powerupsFor(IProtocolFactoryFactory):
            factories.append((f.__class__.__name__.decode('ascii'),
                              f,
                              False))

        f = LiveForm(
            self.portConf.createPort,
            [Parameter('portNumber', TEXT_INPUT, port, 'Port Number',
                       'Integer 0 <= n <= 65535 giving the TCP port to bind.'),

             Parameter('interface', TEXT_INPUT, unicode, 'Interface',
                       'Hostname to bind to, or blank for all interfaces.'),

             Parameter('ssl', CHECKBOX_INPUT, bool, 'SSL',
                       'Select to indicate port should use SSL.'),

             # Text area?  File upload?  What?
             Parameter('certPath', TEXT_INPUT, unicode, 'Certificate Path',
                       'Path to a certificate file on the server, if SSL is to be used.'),

             ChoiceParameter('factory', factories, 'Protocol Factory',
                             'Which pre-existing protocol factory to associate with this port.')])
        f.setFragmentParent(self)
        # f.docFactory = webtheme.getLoader(f.fragmentName)
        return tag[f]