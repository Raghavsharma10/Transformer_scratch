def to_instance(self, response):
        """
        transforms the content to a new instace of
        object self.schema['title']
        :param content: valid response
        :returns new instance of current class
        """
        klass = self.schema['title']
        cls = get_model_class(klass, api=self.__api__)
        jdict = json.loads(response.content, encoding="utf-8")
        ### check if we have a response
        properties_dict = jdict[self.schema['title']]
        # @todo: find a way to handle the data
        # validation fails if the none values are not removed
        new_dict = helpers.remove_properties_containing_None(properties_dict)
        #jdict[self.schema['title']] = new_dict
        obj = cls(new_dict)
        #obj.links = jdict[self.schema['title']]['links']
        return obj