def dict_to_instance(self, content):
        """
        transforms the content to a new instace of
        object self.schema['title']
        :param content: valid response
        :returns new instance of current class
        """
        klass = self.schema['title']
        cls = get_model_class(klass, api=self.__api__)
        # jdict = json.loads(content, encoding="utf-8")
        ### check if we have a response
        properties_dict = content[self.schema['title']][self.schema['title']]
        #@todo: find a way to handle the data
        # validation fails if the none values are not removed
        new_dict = helpers.remove_properties_containing_None(properties_dict)
        obj = cls(new_dict)
        #obj.links = content[self.schema['title']]['links']
        return obj