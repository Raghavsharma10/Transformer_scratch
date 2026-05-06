def make_container_tree(folders,files,path_delim="/",parse_files=True):
    '''make_container_tree will convert a list of folders and files into a json structure that represents a graph.
    :param folders: a list of folders in the image
    :param files: a list of files in the folder
    :param parse_files: return 'files' lookup in result, to associate ID of node with files (default True)
    :param path_delim: the path delimiter, default is '/'
    '''
    nodes = {}  # first we will make a list of nodes
    lookup = {}
    count = 1   # count will hold an id for nodes
    max_depth = 0
    for folder in folders:
        if folder != ".":
            folder = re.sub("^[.]/","",folder)
            path_components = folder.split(path_delim)
            for p in range(len(path_components)):
                path_component = path_components[p]
                fullpath = path_delim.join(path_components[0:p+1])
                # Have we created the node yet?
                if fullpath not in lookup:
                    lookup[fullpath] = count
                    node = {"id":count,"name":path_component,"path":fullpath,"level":p,"children":[]}
                    count +=1
                    # Did we find a deeper level?
                    if p > max_depth:
                        max_depth = p
                    # Does the node have a parent?
                    if p==0: # base node, no parent
                        parent_id = 0
                    else: # look up the parent id
                        parent_path = path_delim.join(path_components[0:p])
                        parent_id = lookup[parent_path]                   
                    node["parent"] = parent_id
                    nodes[node['id']] = node   
           
    # Now make the graph, we simply append children to their parents
    seen = []
    graph = []
    iters = list(range(max_depth+1)) # 0,1,2,3...
    iters.reverse()            # ...3,2,1,0
    iters.pop()                # remove 0
    for level in iters:
        children = {x:y for x,y in nodes.items() if y['level'] == level}
        seen = seen + [y['id'] for x,y in children.items()]
        nodes = {x:y for x,y in nodes.items() if y['id'] not in seen}
        for node_id,child_node in children.items():
            if node_id == 0: #base node
                graph[node_id] = child_node
            else:
                parent_id = child_node['parent']
                nodes[parent_id]["children"].append(child_node)

    # Now add the parents to graph, with name as main lookup
    for parent,parent_info in nodes.items():
        graph.append(parent_info)
    graph = {"name":"base","children":graph}
    result = {"graph":graph,"lookup":lookup,"depth":max_depth+1}

    # Parse files to include in tree
    if parse_files == True:
        file_lookup = {}
        for filey in files:
            filey = re.sub("^[.]/","",filey)
            filepath,filename = os.path.split(filey)
            if filepath in lookup:
                folder_id = lookup[filepath]
                if folder_id in file_lookup:
                    file_lookup[folder_id].append(filename)
                else:
                    file_lookup[folder_id] = [filename]
            elif filepath == '': # base folder
                if 0 in file_lookup:
                    file_lookup[0].append(filename)
                else:
                    file_lookup[0] = [filename]
        result['files'] = file_lookup

    return result