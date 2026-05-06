def get_page(self, form):
        '''Get the requested page'''
        page_size = form.cleaned_data['iDisplayLength']
        start_index = form.cleaned_data['iDisplayStart']
        paginator = Paginator(self.object_list, page_size)
        num_page = (start_index / page_size) + 1
        return paginator.page(num_page)