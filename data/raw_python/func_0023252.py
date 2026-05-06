def _update_method(self, view):
        """Decide which method to use for *view* and configure it accordingly.
        """
        method = self._method
        if method == 'auto':
            if view.transforms.get_transform().Linear:
                method = 'subdivide'
            else:
                method = 'impostor'
        view._method_used = method

        if method == 'subdivide':
            view.view_program['method'] = 0
            view.view_program['a_position'] = self._subdiv_position
            view.view_program['a_texcoord'] = self._subdiv_texcoord
        elif method == 'impostor':
            view.view_program['method'] = 1
            view.view_program['a_position'] = self._impostor_coords
            view.view_program['a_texcoord'] = self._impostor_coords
        else:
            raise ValueError("Unknown image draw method '%s'" % method)

        self.shared_program['image_size'] = self.size
        view._need_method_update = False
        self._prepare_transforms(view)