def getOutputBlocks():
    """return a dict with the output of each function"""
    raw=open("output.txt").read()
    d={}
    for block in raw.split("\n####### ")[1:]:
        title=block.split("\n")[0].split("(")[0]
        block=block.split("\n",1)[1].strip()
        d[title]=block.split("\nfinished in ")[0]
    return d