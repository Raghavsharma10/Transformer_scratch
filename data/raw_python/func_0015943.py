def render_to_response(self, form, **kwargs):
        '''Render Datatables expected JSON format'''
        page = self.get_page(form)
        data = {
            'iTotalRecords': page.paginator.count,
            'iTotalDisplayRecords': page.paginator.count,
            'sEcho': form.cleaned_data['sEcho'],
            'aaData': self.get_rows(page.object_list),
        }
        return self.json_response(data)