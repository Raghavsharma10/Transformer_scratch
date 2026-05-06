def create_image_base64():
    """
        创建一个验证码 + 图片base64字符串流
    """
    code, file = create_image()
    b64str = "data:image/png;base64,"
    with open(file, "rb") as f:
        if sys.version_info.major == 2:
            b64str += base64.b64encode(f.read())
        else:
            b64str += base64.b64encode(f.read()).decode()
    os.remove(file)
    return True, {"code": code, "base64": b64str}