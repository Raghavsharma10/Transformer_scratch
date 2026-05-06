def return_net(word,word_net,depth=1):
    '''Creates a list of unique words that are at a provided depth from root word.

    @Args:
    --
    word :      root word from which the linked words should be returned.
    word_net :  word network (dictionary of word instances)to be refered in this process.
    depth :     depth to which this process must traverse and return words.

    @return:
    ---
    res :   List of words that are within a certain depth from root word in network.

    '''
    if depth<1: raise Exception(TAG+"Degree value error.range(1,~)")
    if depth==1:
        return list(word_net[word].frwrd_links)
    elif depth>1:
        words  = word_net[word].frwrd_links
        res=[]
        for w in words: res.extend(return_net(w,word_net,depth=depth-1))
        return list(set(res))