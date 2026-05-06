def post(self, *args, **kwargs):
        """Save file and return saved info or report errors."""
        if self.upload_allowed():
            form = self.get_upload_form()
            result = {}
            if form.is_valid():
                storage = self.get_storage()
                result['is_valid'] = True
                info = form.stash(storage, self.request.path)
                result.update(info)
            else:
                result.update({
                    'is_valid': False,
                    'errors': form.errors,
                })
            return HttpResponse(json.dumps(result), content_type='application/json')
        else:
            return HttpResponseForbidden()