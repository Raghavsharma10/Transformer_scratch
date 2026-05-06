def build_parameters(request, meta, orgaMode, currentOrga):
    """Return the list of get, post and file parameters to send"""

    postParameters = {}
    getParameters = {}
    files = {}

    def update_parameters(data):
        tmp_getParameters, tmp_postParameters, tmp_files = data

        getParameters.update(tmp_getParameters)
        postParameters.update(tmp_postParameters)
        files.update(tmp_files)

    update_parameters(build_base_parameters(request))
    update_parameters(build_user_requested_parameters(request, meta))
    update_parameters(build_orga_parameters(request, orgaMode, currentOrga))

    return (getParameters, postParameters, files)