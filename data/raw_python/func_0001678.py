def seqfy(strs):
    ''' 序列化 字符串--->实际效果是，为字符串，添加行号，返回字符串
    Sampe usage:
        strs = ["", None, u"First-line\nSecond-line\nThird-line", u"没有换行符"]
        for s in strs:
            print "---"
            result = seqfy(s)
            print result
            print unseqfy(result)
    '''
    
    if not strs:
        return
    
    result = ""
    seq = 1
    ss = strs.split("\n")
    for i in ss:
        if i:
            result = "".join([result, str(seq), ".", i, "\n"])
            seq = seq + 1            
    return result