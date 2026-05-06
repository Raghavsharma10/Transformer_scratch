def _get_name(self):
        """
        There are three cases, because apipie definitions can have multiple
        signatures but python does not
        For example, the api endpoint:
           /api/myres/:myres_id/subres/:subres_id/subres2

        for method *index* will be translated to the api method name:
            subres_index_subres2

        So when you want to call it from v2 object, you'll have:

          myres.subres_index_subres2

        """
        if self.url.count(':') > 1:
            # /api/one/two/:three/four -> two_:three_four
            base_name = self.url.split('/', 3)[-1].replace('/', '_')[1:]
            # :one_two_three -> two_three
            if base_name.startswith(':'):
                base_name = base_name.split('_')[-1]
            # one_:two_three_:four_five -> one_three_five
            base_name = re.sub('_:[^/]+', '', base_name)
            # in case that the last term was a parameter
            if base_name.endswith('_'):
                base_name = base_name[:-1]
            # one_two_three -> one_two_method_three
            base_name = (
                '_' + self._method['name']
            ).join(base_name.rsplit('_', 1))
        else:
            base_name = self._method['name']
        if base_name == 'import':
            base_name = 'import_'
        if self._apipie_resource != self.resource:
            return '%s_%s' % (self._apipie_resource, base_name)
        else:
            return base_name