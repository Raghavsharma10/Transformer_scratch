def addbr(name):
    ''' Create new bridge with the given name '''
    fcntl.ioctl(ifconfig.sockfd, SIOCBRADDBR, name)
    return Bridge(name)