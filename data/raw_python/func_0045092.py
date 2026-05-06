def _print_links(self, model, links):
        """
        Print links that start from model.
        """
        for link in links:
            if link['o2o'] is True:
                link_type = self._one_to_one
            elif link['m2m'] is True:
                link_type = self._many_to_many
            else:
                link_type = self._one_to_many
            linked_model = link['mdl'](super_context)
            self._print('%s %s %s' % (model.title, link_type, linked_model.title))