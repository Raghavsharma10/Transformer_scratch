def find_knn(tf_idf,input_word,k=WIN,rand_on=True):
    '''func that find k nearest neighbors of a spcified word from a given file of tf-idf values of words for list of docs.
    @Args:
    --
    tfidf_file_name : File name for the .tfidfpkl file to be used for searching.
    input_word : word for which process need to be done.
    k : amount of nearest neighbors to yield.(default=cf.nn_window)

    @yields:
    word : kth Nearnest Neighbor of the provided input_word for specified file.(generator yield)
    '''
    
    #find docs that have input_word and gather their content
    word_bag = {}
    for doc in tf_idf:
        contains_flag = False
        #print(doc)
        if input_word in doc.keys():
            #this code will only generate unique words and their tf_idf values, overwritten when already available..
            word_bag.update(doc)    
    #sort the available list of words from docs according to their tf_idf values.
    word_bag = sorted(word_bag.items(), key=operator.itemgetter(1))
    #reverse the order as to get words with large TF-IDF values(descending order)
    word_bag.reverse()
    for element in word_bag[:k]:
        #condition where the word_bag throws the same word as input...(highly likely, hence ignored)
        if element[0] == input_word or (bool(random.getrandbits(1)) and rand_on): 
            continue # randomly select if current word will get yielded or to continue to next word.
        #creating a generator structure for better efficiency of code.
        yield element[0]