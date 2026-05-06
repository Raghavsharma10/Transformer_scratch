def is_root_owned(member):
    '''assess if a file is root owned, meaning "root" or user/group 
    id of 0'''
    if member.uid == 0 or member.gid == 0:
        return True
    elif member.uname == 'root' or member.gname == 'root':
        return True
    return False