def build_base_parameters(request):
    """Build the list of parameters to forward from the post and get parameters"""

    getParameters = {}
    postParameters = {}
    files = {}

    # Copy GET parameters, excluding ebuio_*
    for v in request.GET:
        if v[:6] != 'ebuio_':
            val = request.GET.getlist(v)

            if len(val) == 1:
                getParameters[v] = val[0]
            else:
                getParameters[v] = val

    # If using post, copy post parameters and files. Excluding ebuio_*
    if request.method == 'POST':
        for v in request.POST:
            if v[:6] != 'ebuio_':
                val = request.POST.getlist(v)

                if len(val) == 1:
                    postParameters[v] = val[0]
                else:
                    postParameters[v] = val

        for v in request.FILES:
            if v[:6] != 'ebuio_':
                files[v] = request.FILES[v]  # .chunks()

    return (getParameters, postParameters, files)