def post_data(page, username, password):
    """
    Given username and password, return the post data necessary for login
    """
    soup = BeautifulSoup(page)
    try:
        inputs = soup.find(id='hiddens').findAll('input')
        post_data = {input['name']: input['value'] for input in inputs}
        post_data['username'] = username
        post_data['passwd'] = password
        return post_data
    except:
        return None