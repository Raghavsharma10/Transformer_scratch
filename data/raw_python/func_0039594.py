def validate(self, data):
        """Apply a JSON schema to an object"""
        try:
            schema_path = os.path.normpath(SCHEMA_ROOT)
            location = u'file://%s' % (schema_path)
            fs_resolver = resolver.LocalRefResolver(location, self.schema)
            jsonschema.Draft3Validator(self.schema, resolver=fs_resolver).validate(data)

        except jsonschema.ValidationError as exc:
            # print "data %s" % (data)
            raise jsonschema.exceptions.ValidationError(str(exc))