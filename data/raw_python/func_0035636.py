def _loadDatabase(self, databaseLocation, stream=False):
        """ Loads the database from a given file path in the class

        :param databaseLocation: the location on disk or the stream object
        :param stream: if true treats the databaseLocation as a stream object
        """

        # Initialise Database
        self.systems = []
        self.binaries = []
        self.stars = []
        self.planets = []

        if stream:
            tree = ET.parse(databaseLocation)
            for system in tree.findall(".//system"):
                self._loadSystem(system)
        else:
            databaseXML = glob.glob(os.path.join(databaseLocation, '*.xml'))
            if not len(databaseXML):
                raise LoadDataBaseError('could not find the database xml files. Have you given the correct location '
                                        'to the open exoplanet catalogues /systems folder?')

            for filename in databaseXML:
                try:
                    with open(filename, 'r') as f:
                        tree = ET.parse(f)
                except ET.ParseError as e:  # this is sometimes raised rather than the root.tag system check
                    raise LoadDataBaseError(e)

                root = tree.getroot()

                # Process the system
                if not root.tag == 'system':
                    raise LoadDataBaseError('file {0} does not contain a valid system - could be an error with your version'
                                            ' of the catalogue'.format(filename))

                self._loadSystem(root)