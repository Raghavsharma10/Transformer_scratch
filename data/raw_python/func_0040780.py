def parsePermission3Char(permission):
    """
    'rwx' 形式のアクセス権限文字列 permission を8進数形式に変換する

    :return:
    :rtype: int
    """

    if len(permission) != 3:
        raise ValueError(permission)

    permission_int = 0
    if permission[0] == "r":
        permission_int += 4
    if permission[1] == "w":
        permission_int += 2
    if permission[2] == "x":
        permission_int += 1

    return permission_int