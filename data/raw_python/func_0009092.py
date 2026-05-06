def extract_content(image_path, member_name, return_hash=False):
    '''extract_content will extract content from an image using cat.
    If hash=True, a hash sum is returned instead
    '''
    if member_name.startswith('./'):
        member_name = member_name.replace('.','',1)
    if return_hash:
        hashy = hashlib.md5()

    try:
        content = Client.execute(image_path,'cat %s' %(member_name))
    except:
        return None

    if not isinstance(content,bytes):
        content = content.encode('utf-8')
        content = bytes(content)

    # If permissions don't allow read, return None
    if len(content) == 0:
        return None
    if return_hash:
        hashy.update(content)
        return hashy.hexdigest()
    return content