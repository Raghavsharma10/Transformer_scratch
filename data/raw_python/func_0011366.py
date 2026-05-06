def get_post_fields(request):
    '''parse through a request, and return fields from post in a dictionary
    '''
    fields = dict()
    for field,value in request.form.items():
        fields[field] = value
    return fields