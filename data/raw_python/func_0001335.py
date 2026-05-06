def gen_vocab(cli, args):
    ''' Generate vocabulary list from a tokenized file '''
    if args.topk and args.topk <= 0:
        topk = None
        cli.logger.warning("Invalid k will be ignored (k should be greater than or equal to 1)")
    else:
        topk = args.topk
    if args.stopwords:
        with open(args.stopwords, 'r') as swfile:
            stopwords = swfile.read().splitlines()
    else:
        stopwords = []
    if os.path.isfile(args.input):
        cli.logger.info("Generating vocabulary list from file {}".format(args.input))
        with codecs.open(args.input, encoding='utf-8') as infile:
            if args.output:
                cli.logger.info("Output: {}".format(args.output))
            rp = TextReport(args.output)
            lines = infile.read().splitlines()
            c = Counter()
            for line in lines:
                words = line.split()
                c.update(w for w in words if w not in stopwords)
            # report vocab
            word_freq = c.most_common(topk)
            words = [k for k, v in word_freq]
            rp.header("Lexicon")
            rp.writeline("\n".join(textwrap.wrap(" ".join(w for w in words), width=70)))
            for k, v in word_freq:
                rp.print("{}: {}".format(k, v))
    else:
        cli.logger.warning("File {} does not exist".format(args.input))