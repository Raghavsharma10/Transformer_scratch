def delete_field(self,collection,field_name):
        '''
        Deletes a field from the Solr Collection. Will raise ValueError if the field doesn't exist.

        :param string collection: Name of the collection for the action
        :param string field_name: String name of the field.
        '''
        if not self.does_field_exist(collection,field_name):
            raise ValueError("Field {} Doesn't Exists in Solr Collection {}".format(field_name,collection))
        else:
            temp = {"delete-field" : { "name":field_name }}
            res, con_info = self.solr.transport.send_request(method='POST',endpoint=self.schema_endpoint,collection=collection, data=json.dumps(temp))
            return res