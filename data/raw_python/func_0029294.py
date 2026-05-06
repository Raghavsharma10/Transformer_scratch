def _render_response(self, value, system):
        """ Render a response """
        view = system['view']
        enc_class = getattr(view, '_json_encoder', None)
        if enc_class is None:
            enc_class = get_json_encoder()
        return json.dumps(value, cls=enc_class)