def _cut(ver):
    '''Cuts the version to array, excepts valid version'''
    ver = ver.split('.')
    for i, part in enumerate(ver):
        try:
            ver[i] = int(part)
        except:
            if part[-len('dev'):] == 'dev':
                ver[i] = int(part[:-len('dev')])
                ver.append(-3)
            else:
                ver[i] = int(part[:-len('a')])
                if part[-len('a'):] == 'a':
                    ver.append(-2)
                else:
                    ver.append(-1)
    return ver