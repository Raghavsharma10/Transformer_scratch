def find_tf_idf(file_names=['./../test/testdata'],prev_file_path=None, dump_path=None):
    '''Function to create a TF-IDF list of dictionaries for a corpus of docs.
    If you opt for dumping the data, you can provide a file_path with .tfidfpkl extension(standard made for better understanding)
    and also re-generate a new tfidf list which overrides over an old one by mentioning its path.

    @Args:
    --
    file_names :      paths of files to be processed on, these files are created using twitter_streaming module.
    prev_file_path :  path of old .tfidfpkl file, if available. (default=None)
    dump_path :       directory-path where to dump generated lists.(default=None)

    @returns:
    --
    df :     a dict of unique words in corpus,with their document frequency as values.
    tf_idf : the generated tf-idf list of dictionaries for mentioned docs.
    '''
    tf_idf = [] # will hold a dict of word_count for every doc(line in a doc in this case)
    df = defaultdict(int)
    # this statement is useful for altering existant tf-idf file and adding new docs in itself.(## memory is now the biggest issue)
    if prev_file_path:
        print(TAG,'modifying over exising file.. @',prev_file_path)
        df,tf_idf = pickle.load(open(prev_file_path,'rb'))
        prev_doc_count = len(df)
        prev_corpus_length = len(tf_idf)

    for f in file_names:
        # never use 'rb' for textual data, it creates something like,  {b'line-inside-the-doc'}
        with open(f,'r') as file1:
            #create word_count dict for all docs
            for line in file1:
                wdict = defaultdict(int)
                #find the amount of doc a word is in
                for word in set(line.split()):
                    df[word] +=1
                #find the count of all words in every doc
                for word in line.split():
                    wdict[word] += 1
                tf_idf.append(wdict)

    #calculating final TF-IDF values  for all words in all docs(line is a doc in this case)
    for doc in tf_idf:
        for key in doc:
            true_idf = math.log(len(tf_idf)/df[key])
            true_tf = doc[key]/float(len(doc))
            doc[key] = true_tf * true_idf

    print(TAG,'Total number of unique words in corpus',len(df),'( '+paint('++'+str(len(df)-prev_doc_count),'g')+' )' if prev_file_path else '')
    print(TAG,'Total number of docs in corpus:',len(tf_idf),'( '+paint('++'+str(len(tf_idf)-prev_corpus_length),'g')+' )' if prev_file_path else '')
    
    # dump if a dir-path is given
    if dump_path:
        if dump_path[-8:] == 'tfidfpkl': 
            pickle.dump((df,tf_idf),open(dump_path,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
            print(TAG,'Dumping TF-IDF vars @',dump_path)
    return df,tf_idf