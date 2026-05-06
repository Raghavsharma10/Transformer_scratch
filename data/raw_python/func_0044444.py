def initMazeFromJSON(jsonString, cellClass=Cell, gridClass=Grid):
    '''Init a maze from JSON string.'''
    
    jsonObj = json.loads(jsonString)
    rows=jsonObj["rows"]
    columns=jsonObj["columns"]
    grid=gridClass(rows,columns,cellClass)
    grid.algorithm=jsonObj["algorithm"]
    grid.algorithm_key=jsonObj["algorithm_key"]
    grid.braid=jsonObj["braid"]

    #init cells
    #for each cell link those that are neigbors
    for _cell in jsonObj["cells"]:
        cell=json.loads(_cell)
        gridCell=grid.getCell(cell["row"],cell["column"])
        if "content" in cell:
            gridCell.content=cell["content"]
        else:
            gridCell.setContent(" ")
        
        if cell["north"]:
            gridCell.link(gridCell.north)
        if cell["east"]:
            gridCell.link(gridCell.east)
        if cell["south"]:
            gridCell.link(gridCell.south)
        if cell["west"]:
            gridCell.link(gridCell.west)

    return grid