def delete(self, request, id):
        """ Remove the targeted feature from the database """
        if self.readonly:
            return HTTPMethodNotAllowed(headers={'Allow': 'GET, HEAD'})
        session = self.Session()
        obj = session.query(self.mapped_class).get(id)
        if obj is None:
            return HTTPNotFound()
        if self.before_delete is not None:
            self.before_delete(request, obj)
        session.delete(obj)
        return Response(status_int=204)