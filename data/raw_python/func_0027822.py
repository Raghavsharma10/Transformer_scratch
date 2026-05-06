def getCodeBlocks():
    """return a dict with the code for each function"""
    raw=open("examples.py").read()
    d={}
    for block in raw.split("if __name__")[0].split("\ndef "):
        title=block.split("\n")[0].split("(")[0]
        if not title.startswith("demo_"):
            continue
        code=[x[4:] for x in block.split("\n")[1:] if x.startswith("    ")]
        d[title]="\n".join(code).strip()
    return d