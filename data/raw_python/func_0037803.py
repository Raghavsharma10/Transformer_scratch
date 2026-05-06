def select_token(request, scopes='', new=False):
    """
    Presents the user with a selection of applicable tokens for the requested view.
    """

    @tokens_required(scopes=scopes, new=new)
    def _token_list(r, tokens):
        context = {
            'tokens': tokens,
            'base_template': app_settings.ESI_BASE_TEMPLATE,
        }
        return render(r, 'esi/select_token.html', context=context)

    return _token_list(request)