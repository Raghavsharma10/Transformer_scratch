def get_base_url(html: str) -> str:
    """
    Search for login url from VK login page
    """
    forms = BeautifulSoup(html, 'html.parser').find_all('form')
    if not forms:
        raise VVKBaseUrlException('Form for login not found')
    elif len(forms) > 1:
        raise VVKBaseUrlException('More than one login form found')
    login_url = forms[0].get('action')
    if not login_url:
        raise VVKBaseUrlException('No action tag in form')
    return login_url