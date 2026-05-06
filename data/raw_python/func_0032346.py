def upgradeUserInfo1to2(oldUserInfo):
    """
    Concatenate the I{firstName} and I{lastName} attributes from the old user
    info item and set the result as the I{realName} attribute of the upgraded
    item.
    """
    newUserInfo = oldUserInfo.upgradeVersion(
        UserInfo.typeName, 1, 2,
        realName=oldUserInfo.firstName + u" " + oldUserInfo.lastName)
    return newUserInfo