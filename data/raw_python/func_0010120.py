def get_parser():
    """Parse command-line arguments."""
    parser = ArgumentParser(description='a command-line web scraping tool')
    parser.add_argument('query', metavar='QUERY', type=str, nargs='*',
                        help='URLs/files to scrape')
    parser.add_argument('-a', '--attributes', type=str, nargs='*',
                        help='extract text using tag attributes')
    parser.add_argument('-all', '--crawl-all', help='crawl all pages',
                        action='store_true')
    parser.add_argument('-c', '--crawl', type=str, nargs='*',
                        help='regexp rules for following new pages')
    parser.add_argument('-C', '--clear-cache', help='clear requests cache',
                        action='store_true')
    parser.add_argument('--csv', help='write files as csv',
                        action='store_true')
    parser.add_argument('-cs', '--cache-size', type=int, nargs='?',
                        help='size of page cache (default: 1000)',
                        default=1000)
    parser.add_argument('-f', '--filter', type=str, nargs='*',
                        help='regexp rules for filtering text')
    parser.add_argument('--html', help='write files as HTML',
                        action='store_true')
    parser.add_argument('-i', '--images', action='store_true',
                        help='save page images')
    parser.add_argument('-m', '--multiple', help='save to multiple files',
                        action='store_true')
    parser.add_argument('-max', '--max-crawls', type=int,
                        help='max number of pages to crawl')
    parser.add_argument('-n', '--nonstrict', action='store_true',
                        help='allow crawler to visit any domain')
    parser.add_argument('-ni', '--no-images', action='store_true',
                        help='do not save page images')
    parser.add_argument('-no', '--no-overwrite', action='store_true',
                        help='do not overwrite files if they exist')
    parser.add_argument('-o', '--out', type=str, nargs='*',
                        help='specify outfile names')
    parser.add_argument('-ow', '--overwrite', action='store_true',
                        help='overwrite a file if it exists')
    parser.add_argument('-p', '--pdf', help='write files as pdf',
                        action='store_true')
    parser.add_argument('-pt', '--print', help='print text output',
                        action='store_true')
    parser.add_argument('-q', '--quiet', help='suppress program output',
                        action='store_true')
    parser.add_argument('-s', '--single', help='save to a single file',
                        action='store_true')
    parser.add_argument('-t', '--text', help='write files as text',
                        action='store_true')
    parser.add_argument('-v', '--version', help='display current version',
                        action='store_true')
    parser.add_argument('-x', '--xpath', type=str, nargs='?',
                        help='filter HTML using XPath')
    return parser