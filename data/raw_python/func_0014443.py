def create_copy_field(self,collection,copy_dict):
        '''
        Creates a copy field.

        copy_dict should look like ::

            {'source':'source_field_name','dest':'destination_field_name'}

        :param string collection: Name of the collection for the action
        :param dict copy_field: Dictionary of field info

        Reference: https://cwiki.apache.org/confluence/display/solr/Schema+API#SchemaAPI-AddaNewCopyFieldRule
        '''
        temp = {"add-copy-field":dict(copy_dict)}
        res, con_info = self.solr.transport.send_request(method='POST',endpoint=self.schema_endpoint,collection=collection, data=json.dumps(temp))
        return res