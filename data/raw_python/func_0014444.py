def delete_copy_field(self, collection, copy_dict):
        '''
        Deletes a copy field.

        copy_dict should look like ::

            {'source':'source_field_name','dest':'destination_field_name'}

        :param string collection: Name of the collection for the action
        :param dict copy_field: Dictionary of field info
        '''

        #Fix this later to check for field before sending a delete
        if self.devel:
            self.logger.debug("Deleting {}".format(str(copy_dict)))
        copyfields = self.get_schema_copyfields(collection)
        if copy_dict not in copyfields:
            self.logger.info("Fieldset not in Solr Copy Fields: {}".format(str(copy_dict)))
        temp = {"delete-copy-field": dict(copy_dict)}
        res, con_info = self.solr.transport.send_request(method='POST',endpoint=self.schema_endpoint,collection=collection, data=json.dumps(temp))
        return res