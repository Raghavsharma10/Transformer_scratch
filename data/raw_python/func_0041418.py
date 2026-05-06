def log(fname,msg):
    ''' generic logging function '''
    with open(fname,'a') as f:
        f.write(datetime.datetime.now().strftime('%m-%d-%Y %H:%M:\n') + msg + '\n')