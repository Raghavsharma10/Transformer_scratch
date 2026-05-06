def main():
    ''' ChirpText Tools main function '''
    app = CLIApp(desc='ChirpText Tools', logger=__name__, show_version=show_version)
    # add tasks
    vocab_task = app.add_task('vocab', func=gen_vocab)
    vocab_task.add_argument('input', help='Input file')
    vocab_task.add_argument('--output', help='Output file', default=None)
    vocab_task.add_argument('--stopwords', help='Stop word to ignore', default=None)
    vocab_task.add_argument('-k', '--topk', help='Only select the top k frequent elements', default=None, type=int)
    # run app
    app.run()