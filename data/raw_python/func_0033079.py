def create_from_json(cls, json_str, ignore_non_defaults=True):
        """
        Creates a database object from a json object.  The intent of this method
        is to allow creating a database object directly from json.
        
        Mongolia will also automatically convert any json values that are
        formatted using the MongoliaJSONEncoder (for ObjectIds and datetime
        objects) back to their native python data types.
        
        Note: if using AngularJS, make sure to pass json back using
        `angular.toJson(obj)` instead of `JSON.stringify(obj)` since angular
        sometimes adds `$$hashkey` to javascript objects and this will cause
        a mongo error due to the "$" prefix in keys.
        
        @param json_str: the json string containing the new object to use for
            creating the new object
        @param ignore_non_defaults: if this is True and the database object
            has non-empty DEFAULTS, then any top-level keys of the create json
            that do not appear in DEFAULTS will also be excluded in creation
        """
        create_dict = json.loads(json_str, cls=MongoliaJSONDecoder, encoding="utf-8")
        # Remove all keys not in DEFAULTS if ignore_non_defaults is True
        if cls.DEFAULTS and ignore_non_defaults:
            for key in frozenset(create_dict).difference(frozenset(cls.DEFAULTS)):
                del create_dict[key]
        
        cls.create(create_dict, random_id=True)