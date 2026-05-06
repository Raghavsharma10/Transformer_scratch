def json_update(self, json_str, exclude=[], ignore_non_defaults=True):
        """
        Updates a database object based on a json object.  The intent of this
        method is to allow passing json to an interface which then subsequently
        manipulates the object and then sends back an update.
        
        Mongolia will also automatically convert any json values that were
        initially converted from ObjectId and datetime.datetime objects back
        to their native python object types.
        
        Note: if using AngularJS, make sure to pass json back using
        `angular.toJson(obj)` instead of `JSON.stringify(obj)` since angular
        sometimes adds `$$hashkey` to javascript objects and this will cause
        a mongo error due to the "$" prefix in keys.
        
        @param json_str: the json string containing the new object to use for
            the update
        @param exclude: a list of top-level keys to exclude from the update
            (ID_KEY need not be included in this list; it is automatically
            deleted since it can't be part of a mongo update operation)
        @param ignore_non_defaults: if this is True and the database object
            has non-empty DEFAULTS, then any top-level keys in the update json
            that do not appear in DEFAULTS will also be excluded from the update
        """
        update_dict = json.loads(json_str, cls=MongoliaJSONDecoder, encoding="utf-8")
        # Remove ID_KEY since it can't be part of a mongo update operation
        if ID_KEY in update_dict:
            del update_dict[ID_KEY]
        
        # Remove all keys in the exclude list from the update
        for key in frozenset(exclude).intersection(frozenset(update_dict)):
            del update_dict[key]
        
        # Remove all keys not in DEFAULTS if ignore_non_defaults is True
        if self.DEFAULTS and ignore_non_defaults:
            for key in frozenset(update_dict).difference(frozenset(self.DEFAULTS)):
                del update_dict[key]
        
        self.update(update_dict)