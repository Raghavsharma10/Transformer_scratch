def name(self):
        """ access to classic name attribute is hidden by this property """
        return self.NAME_SEPARATOR.join([super(TaggedVertex, self).name] + self.get_tags_as_list_of_strings())