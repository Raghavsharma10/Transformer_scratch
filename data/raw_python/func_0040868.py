def check_page_for_warnings(html: str) -> None:
    """
    Checks if is any warnings on page if so raises an exception
    """
    soup = BeautifulSoup(html, 'html.parser')
    warnings = soup.find_all('div', {'class': 'service_msg_warning'})
    if warnings:
        exception_msg = '; '.join((warning.get_text() for warning in warnings))
        raise VVKPageWarningException(exception_msg)