def create_field(self, collection, field_dict):
        '''
        Creates a new field in managed schema, will raise ValueError if the field already exists.  field_dict should look like this::

            {
                 "name":"sell-by",
                 "type":"tdate",
                 "stored":True
            }

        Reference: https://cwiki.apache.org/confluence/display/solr/Defining+Fields

        '''
        if self.does_field_exist(collection,field_dict['name']):
            raise ValueError("Field {} Already Exists in Solr Collection {}".format(field_dict['name'],collection))
        temp = {"add-field":dict(field_dict)}
        res, con_info =self.solr.transport.send_request(method='POST',endpoint=self.schema_endpoint,collection=collection, data=json.dumps(temp))
        return res