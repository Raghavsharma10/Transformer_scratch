def stepfy(strs):
    ''' 步骤化 字符串 --->实际效果是, 依据 序列化的字符串，转换为 Step_%s_info 的字典， 返回字典
    Sample usage:
        test_strs = [
        "",
        None,
        u"First-line\nSecond-line\nThird-line",
        u'1.First-line\n2.Second-line\n3.Third-line\n',
        u'3.没有换行符',
        u'3.有换行符\n',
        "asdfasdfsdf",    
        "1.asdfasdfsdf\n2.sodfi",
        "1.1.dfasdfahttp://192.168.1.1sdfsdf2.1.1.1.1\n",
        "dfasdfahttp://192.168.1.1sdfsdf2.1.1.1.1\n",
        ]
        for i in test_strs:
            steps = stepfy(i)
            un = unstepfy(steps)
            print "string: %r" %i
            print "stepfy: %s" %steps
            print "unstepfy: %r\n" %un
    '''
    
    result = {}
    prog_step   = re.compile("^\d+\.")
      
    if not strs:
        return result
      
    raws = strs.split("\n")
    for raw in raws:
        step_num = raws.index(raw) + 1
        raw = prog_step.sub("",raw)       
        if raw:
            result["Step_%s_info" %step_num] = raw
    return result