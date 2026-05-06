def json_update_fields(self, json_str, fields_to_update):
        """
        Updates the specified fields of a database object based on a json
        object. The intent of this method is to allow passing json to an
        interface which then subsequently manipulates the object and then sends
        back an update for specific fields of the object.
        
        Mongolia will also automatically convert any json values that were
        initially converted from ObjectId and datetime.datetime objects back
        to their native python object types.
        
        Note: if using AngularJS, make sure to pass json back using
        `angular.toJson(obj)` instead of `JSON.stringify(obj)` since angular
        sometimes adds `$$hashkey` to javascript objects and this will cause
        a mongo error due to the "$" prefix in keys.
        
        @param json_str: the json string containing the new object to use for
            the update
        @param fields_to_update: a list of the top-level keys to update; only
            keys included in this list will be update.  Do not include ID_KEY
            in this list since it can't be part of a mongo update operation
        """
        update_dict = json.loads(json_str, cls=MongoliaJSONDecoder, encoding="utf-8")
        update_dict = dict((k, v) for k, v in update_dict.items()
                       if k in fields_to_update and k != ID_KEY)
        self.update(update_dict)