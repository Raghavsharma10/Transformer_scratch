def resize_program_image(img_url, img_size=300):
    '''
    Resize a program's thumbnail to the desired dimension
    '''
    match = re.match(r'.+/(\d+)x(\d+)/.+', img_url)
    if not match:
        _LOGGER.warning('Could not compute current image resolution of %s',
                        img_url)
        return img_url
    res_x = int(match.group(1))
    res_y = int(match.group(2))
    # aspect_ratio = res_x / res_y
    target_res_y = int(img_size * res_y / res_x)
    return re.sub(
        r'{}x{}'.format(res_x, res_y),
        r'{}x{}'.format(img_size, target_res_y),
        img_url)