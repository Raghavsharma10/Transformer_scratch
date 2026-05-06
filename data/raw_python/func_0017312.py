def parse_command_line():
    """ Parse CLI args."""

    ## create the parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  * Example command-line usage: 

  ## push test branch to conda --label=conda-test for travis CI
  ./versioner.py -p toytree -b test -t 0.1.7 

  ## push master as a new tag to git and conda
  ./versioner.py -p toytree -b master -t 0.1.7 --deploy

  ## build other deps on conda at --label=conda-test
  ./versioner.py -p toyplot --no-git
  ./versioner.py -p pypng --no-git

    """)

    ## add arguments 
    parser.add_argument('-v', '--version', action='version', 
        version="0.1")

    parser.add_argument('-p', #"--package", 
        dest="package", 
        default="toytree",
        type=str, 
        help="the tag to put in __init__ and use on conda")

    parser.add_argument('-b', #"--branch", 
        dest="branch", 
        default="master",
        type=str,
        help="the branch to build conda package from")

    parser.add_argument('-t', #"--tag", 
        dest="tag", 
        default="test",
        type=str, 
        help="the tag to put in __init__ and use on conda")

    parser.add_argument("--deploy", 
        dest="deploy",
        action='store_true',
        help="push the tag to git and upload to conda main label")

    parser.add_argument("--no-git", 
        dest="nogit",
        action='store_true',
        help="skip git update and only build/upload to conda")


    ## if no args then return help message
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    ## parse args
    args = parser.parse_args()
    return args