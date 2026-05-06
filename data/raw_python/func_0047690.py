def xml_import(self,
                   filepath=None,
                   xml_content=None,
                   markings=None,
                   identifier_ns_uri=None,
                   initialize_importer=True,
                   **kwargs):
        """
        Import an OpenIOC indicator xml (root element 'ioc') from file <filepath> or
        from a string <xml_content>

        You can provide:

        - a list of markings with which all generated Information Objects
           will be associated (e.g., in order to provide provenance function)

        - The uri of a namespace of the identifiers for the generated information objects.
          This namespace identifiers the 'owner' of the object. For example, if importing
          IOCs published by Mandiant (e.g., as part of the APT1 report), chose an namespace
          such  as 'mandiant.com' or similar (and be consistent about it, when importing
          other stuff published by Mandiant).

        The kwargs are not read -- they are present to allow the use of the
        DingoImportCommand class for easy definition of commandline import commands
        (the class passes all command line arguments to the xml_import function, so
        without the **kwargs parameter, an error would occur.
        """


        if initialize_importer:
            # Clear state in case xml_import is used several times, but keep namespace info
            self.__init__()

        # Initialize  default arguments

        # '[]' would be mutable, so we initialize here
        if not markings:
            markings = []

        # Initializing here allows us to also get the default namespace when
        # explicitly passing 'None' as parameter.

        if identifier_ns_uri:
            self.identifier_ns_uri = identifier_ns_uri

        # Use the generic XML import customized for  OpenIOC import
        # to turn XML into DingoObjDicts

        import_result =  MantisImporter.xml_import(xml_fname=filepath,
                                                   xml_content=xml_content,
                                                   ns_mapping=self.namespace_dict,
                                                   embedded_predicate=self.openioc_embedding_pred,
                                                   id_and_revision_extractor=self.id_and_revision_extractor,
                                                   transformer=self.transformer,
                                                   keep_attrs_in_created_reference=False,
                                                   )


        # The MANTIS/DINGOS xml importer returns then the following structure::
        #
        #
        #    {'id_and_rev_info': Id and revision info of top-level element of the form
        #        {'id': ... , 'timestamp': ...}
        #    'elt_name': Element name of top-level element
        #    'dict_repr': Dictionary representation of XML, minus the embedded objects -- for
        #                 those, an 'idref' reference has been generated
        #    'embedded_objects': List of embedded objects, as dictionary
        #                       {"id_and_revision_info": id and revision info of extracted object,
        #                        "elt_name": Element name,
        #                        "dict_repr" :  dictionary representation of XML of embedded object
        #                       }
        #    'unprocessed' : List of unprocessed embedded objects (as libxml2 Node object)
        #                    (e.g. for handover to other importer
        #    'file_content': Content of imported file (or, if content was passed instead of a file name,
        #                                                                                         the original content)}


        id_and_rev_info = import_result['id_and_rev_info']
        elt_name = import_result['elt_name']
        elt_dict = import_result['dict_repr']

        embedded_objects = import_result['embedded_objects']

        default_ns = self.namespace_dict.get(elt_dict.get('@@ns',None),'http://schemas.mandiant.com/unknown/ioc')

        # Export family information.
        family_info_dict = search_by_re_list(self.RE_LIST_NS_TYPE_FROM_NS_URL,default_ns)
        if family_info_dict:
            self.iobject_family_name="%s.mandiant.com" % family_info_dict['family']
            self.iobject_family_revision_name=family_info_dict['revision']


        # Initialize stack with import_results.

        # First, the result from the top-level import
        pending_stack = deque()

        pending_stack.append((id_and_rev_info, elt_name,elt_dict))

        # Then the embedded objects
        while embedded_objects:
            embedded_object = embedded_objects.pop()
            id_and_rev_info = embedded_object['id_and_rev_info']
            elt_name = embedded_object['elt_name']
            elt_dict = embedded_object['dict_repr']
            pending_stack.append((id_and_rev_info,elt_name,elt_dict))

        if id_and_rev_info['timestamp']:
            ts = id_and_rev_info['timestamp']
        else:
            ts = self.create_timestamp

        while pending_stack:
            (id_and_rev_info, elt_name, elt_dict) = pending_stack.pop()

            # Call the importer that turns DingoObjDicts into Information Objects in the database
            iobject_type_name = elt_name
            iobject_type_namespace_uri = self.namespace_dict.get(elt_dict.get('@@ns',None),DINGOS_GENERIC_FAMILY_NAME)

            MantisImporter.create_iobject(iobject_family_name = self.iobject_family_name,
                                          iobject_family_revision_name= self.iobject_family_revision_name,
                                          iobject_type_name=iobject_type_name,
                                          iobject_type_namespace_uri=iobject_type_namespace_uri,
                                          iobject_type_revision_name= '',
                                          iobject_data=elt_dict,
                                          uid=id_and_rev_info['id'],
                                          identifier_ns_uri= self.identifier_ns_uri,
                                          timestamp = ts,
                                          create_timestamp = self.create_timestamp,
                                          markings=markings,
                                          config_hooks = {'special_ft_handler' : self.fact_handler_list(),
                                                          'datatype_extractor' : self.datatype_extractor,
                                                          'attr_ignore_predicate' : self.attr_ignore_predicate},
                                          namespace_dict=self.namespace_dict,
                                          )