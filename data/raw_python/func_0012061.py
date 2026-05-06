def bulk(self, data: List[Dict[str, str]], identifier_key: str, op_type='index', upsert=False, keep_id_key=False) -> bool:
        """
        Takes a list of dictionaries and an identifier key and indexes everything into this index.

        :param data:            List of dictionaries containing the data to be indexed.
        :param identifier_key:  The name of the dictionary element which should be used as _id. This will be removed from
                                the body. Is ignored when None or empty string. This will cause elastic
                                to create their own _id.
        :param op_type:         What should be done: 'index', 'delete', 'update'.
        :param upsert:          The update op_type can be upserted, which will create a document if not already present.
        :param keep_id_key      Determines if the value designated as the identifier_key should be kept
                                as part of the document or removed from it.
        :returns                Returns True if all the messages were indexed without errors. False otherwise.
        """
        bulk_objects = []
        for document in data:
            bulk_object = dict()
            bulk_object['_op_type'] = op_type
            if identifier_key is not None and identifier_key != '':
                bulk_object['_id'] = document[identifier_key]
                if not keep_id_key:
                    document.pop(identifier_key)
                if bulk_object['_id'] == '':
                    bulk_object.pop('_id')
            if op_type == 'index':
                bulk_object['_source'] = document
            elif op_type == 'update':
                bulk_object['doc'] = document
                if upsert:
                    bulk_object['doc_as_upsert'] = True
            bulk_objects.append(bulk_object)
            logging.debug(str(bulk_object))
        logging.info('Start bulk index for ' + str(len(bulk_objects)) + ' objects.')
        errors = bulk(self.instance, actions=bulk_objects, index=self.index, doc_type=self.doc_type,
                      raise_on_error=False)
        logging.info(str(errors[0]) + ' documents were successfully indexed/updated/deleted.')
        if errors[0] - len(bulk_objects) != 0:
            logging.error(str(len(bulk_objects) - errors[0]) + ' documents could not be indexed/updated/deleted.')
            for error in errors[1]:
                logging.error(str(error))
            return False
        else:
            logging.debug('Finished bulk %s.', op_type)
            return True