def compile_schema(fields):
        """
        joins schema fields with base solr schema

        :param list[str] fields: field list
        :return: compiled schema
        :rtype: byte
        """
        path = os.path.dirname(os.path.realpath(__file__))
        # path = os.path.dirname(
        #     os.path.abspath(inspect.getfile(inspect.currentframe())))
        with codecs.open("%s/solr_schema_template.xml" % path, 'r', 'utf-8') as fh:
            schema_template = fh.read()

        schema = schema_template.format('\n'.join(fields))
        if six.PY2:
            return schema.encode('utf-8')

        return schema