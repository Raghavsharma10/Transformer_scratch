def reference_handler(self,iobject, fact, attr_info, add_fact_kargs):
        """
        Handler for facts that contain a reference to a fact.
        See below in the comment regarding the fact_handler_list
        for a description of the signature of handler functions.

        As shown below in the handler list, this handler is called
        when a attribute with key '@idref' on the fact's node
        is detected -- this attribute signifies that this fact does not contain
        a value but points to another object. Thus, we try to retrieve that
        object from the database. If it exists, fine -- if not, then the
        call to 'create_iobject' returns a PLACEHOLDER object.

        We further create/refer to the fitting fact data type:
        we want the fact data type to express that the fact is
        a reference to an object.
        """

        (namespace_uri,uid) = (self.identifier_ns_uri,attr_info['idref'])


        # We are always able to extract the timestamp from the referencing node, because for OpenIOC,
        # all references are created by DINGO's generic import, and the import writes the timestamp
        # information into the created reference.

        timestamp = attr_info['@timestamp']

        # The following either retrieves an already existing object of given ID and timestamp
        # or creates a placeholder object.

        (target_mantis_obj, existed) = MantisImporter.create_iobject(
            uid=uid,
            identifier_ns_uri=namespace_uri,
            timestamp=timestamp)

        logger.debug("Creation of Placeholder for %s %s returned %s" % (namespace_uri,uid,existed))

        # What remains to be done is to write the reference to the created placeholder object

        add_fact_kargs['value_iobject_id'] = Identifier.objects.get(uid=uid,namespace__uri=namespace_uri)

        # Handlers have to return 'True', otherwise the fact will not be created.

        return True