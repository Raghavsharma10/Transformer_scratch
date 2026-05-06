def set_mysql_connection(host='localhost', user='pyctd_user', password='pyctd_passwd', db='pyctd', charset='utf8'):
    """Sets the connection using MySQL Parameters"""
    set_connection('mysql+pymysql://{user}:{passwd}@{host}/{db}?charset={charset}'.format(
        host=host,
        user=user,
        passwd=password,
        db=db,
        charset=charset)
    )