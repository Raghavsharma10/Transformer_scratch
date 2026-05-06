def genargs(prog: Optional[str] = None) -> ArgumentParser:
    """
    Create a command line parser
    :return: parser
    """
    parser = ArgumentParser(prog)
    parser.add_argument("rdf", help="Input RDF file or SPARQL endpoint if slurper or sparql options")
    parser.add_argument("shex", help="ShEx specification")
    parser.add_argument("-f", "--format", help="Input RDF Format", default="turtle")
    parser.add_argument("-s", "--start", help="Start shape. If absent use ShEx start node.")
    parser.add_argument("-ut", "--usetype", help="Start shape is rdf:type of focus", action="store_true")
    parser.add_argument("-sp", "--startpredicate", help="Start shape is object of this predicate")
    parser.add_argument("-fn", "--focus", help="RDF focus node")
    parser.add_argument("-A", "--allsubjects", help="Evaluate all non-bnode subjects in the graph", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true", help="Add debug output")
    parser.add_argument("-ss", "--slurper", action="store_true", help="Use SPARQL slurper graph")
    parser.add_argument("-cf", "--flattener", action="store_true", help="Use RDF Collections flattener graph")
    parser.add_argument("-sq", "--sparql", help="SPARQL query to generate focus nodes")
    parser.add_argument("-se", "--stoponerror", help="Stop on an error", action="store_true")
    parser.add_argument("--stopafter", help="Stop after N nodes", type=int)
    parser.add_argument("-ps", "--printsparql", help="Print SPARQL queries as they are executed", action="store_true")
    parser.add_argument("-pr", "--printsparqlresults", help="Print SPARQL query and results", action="store_true")
    parser.add_argument("-gn", "--graphname", help="Specific SPARQL graph to query - use '' for any named graph")
    parser.add_argument("-pb", "--persistbnodes", help="Treat BNodes as persistent in SPARQL endpoint",
                        action="store_true")
    return parser