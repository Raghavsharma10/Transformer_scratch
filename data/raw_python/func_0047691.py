def id_and_revision_extractor(self,xml_elt):
        """
        Function for determing an identifier (and, where applicable, timestamp/revision
        information) for extracted embedded content;
        to be used for DINGO's xml-import hook 'id_and_revision_extractor'.

        This function is called

        - for the top-level node of the XML to be imported.

        - for each node at which an embedded object is extracted from the XML
          (when this occurs is governed by the following function, the
          embedding_pred

        It must return an identifier and, where applicable, a revision and or timestamp;
        in the form of a dictionary {'id':<identifier>, 'timestamp': <timestamp>}.
        How you format the identifier is up to you, because you will have to adopt
        the code in function xml_import such that the Information Objects
        are created with the proper identifier (consisting of qualifying namespace
        and uri.)

        In OpenIOC, the identifier is contained in the 'id' attribute of an element;
        the top-level 'ioc' element carries a timestamp in the 'last-modified' attribute.

        Note: the xml_elt is an XMLNode defined by the Python libxml2 bindings. If you
        have never worked with these, have a look at

        - Mike Kneller's brief intro: http://mikekneller.com/kb/python/libxml2python/part1
        - the functions in django-dingos core.xml_utils module

        """

        result = {'id':None,
                  'timestamp': None}

        attributes = extract_attributes(xml_elt,prefix_key_char='@')

        # Extract identifier:
        if '@id' in attributes:
            result['id']=attributes['@id']

        # Extract time-stamp

        if '@last-modified' in attributes:
            naive = parse_datetime(attributes['@last-modified'].strip())
            if naive:
                # Make sure that information regarding the timezone is
                # included in the time stamp. If it is not, we chose
                # utc as default timezone: if we assume that the same
                # producer of OpenIOC data always uses the same timezone
                # for filling in the 'last-modified' attribute, then
                # this serves the main purpose of time stamps for our
                # means: we can find out the latest revision of a
                # given piece of data.
                if not timezone.is_aware(naive):
                    aware = timezone.make_aware(naive,timezone.utc)
                else:
                    aware = naive
                result['timestamp']= aware

        return result