def _get_order_by(self, request):
        """ Return an SA order_by """
        attr = request.params.get('sort', request.params.get('order_by'))
        if attr is None or not hasattr(self.mapped_class, attr):
            return None
        if request.params.get('dir', '').upper() == 'DESC':
            return desc(getattr(self.mapped_class, attr))
        else:
            return asc(getattr(self.mapped_class, attr))