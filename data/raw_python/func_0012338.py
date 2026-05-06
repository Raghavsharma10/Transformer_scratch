def extract_program_summary(data):
    '''
    Extract the summary data from a program's detail page
    '''
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data, 'html.parser')
    try:
        return soup.find(
            'div', {'class': 'episode-synopsis'}
        ).find_all('div')[-1].text.strip()
    except Exception:
        _LOGGER.info('No summary found for program: %s',
                     soup.find('a', {'class': 'prog_name'}))
        return "No summary"