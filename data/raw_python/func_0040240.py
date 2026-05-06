def triplifyGML(dpath="../data/fb/",fname="foo.gdf",fnamei="foo_interaction.gdf",
        fpath="./fb/",scriptpath=None,uid=None,sid=None,fb_link=None,ego=True,umbrella_dir=None):
    """Produce a linked data publication tree from a standard GML file.

    INPUTS:
    ======
    => the data directory path
    => the file name (fname) of the friendship network
    => the file name (fnamei) of the interaction network
    => the final path (fpath) for the tree of files to be created
    => a path to the script that is calling this function (scriptpath)
    => the numeric id (uid) of the facebook user or group of the network(s)
    => the numeric id (sid) of the facebook user or group of the network (s)
    => the facebook link (fb_link) of the user or group
    => the network is from a user (ego==True) or a group (ego==False)

    OUTPUTS:
    =======
    the tree in the directory fpath."""
    c("iniciado tripgml")
    if sum(c.isdigit() for c in fname)==4:
        year=re.findall(r".*(\d\d\d\d).gml",fname)[0][0]
        B.datetime_snapshot=datetime.date(*[int(i) for i in (year)])
    if sum(c.isdigit() for c in fname)==12:
        day,month,year,hour,minute=re.findall(r".*(\d\d)(\d\d)(\d\d\d\d)_(\d\d)(\d\d).gml",fname)[0]
        B.datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day,hour,minute)])
    if sum(c.isdigit() for c in fname)==14:
        day,month,year,hour,minute,second=re.findall(r".*(\d\d)(\d\d)(\d\d\d\d)_(\d\d)(\d\d)(\d\d).gml",fname)[0]
        B.datetime_snapshot=datetime.datetime(*[int(i) for i in (year,month,day,hour,minute,second)])
    elif sum(c.isdigit() for c in fname)==8:
        day,month,year=re.findall(r".*(\d\d)(\d\d)(\d\d\d\d).gml",fname)[0]
        B.datetime_snapshot=datetime.date(*[int(i) for i in (year,month,day)])
    B.datetime_snapshot_=datetime_snapshot.isoformat()
    B.fname=fname
    B.fnamei=fnamei
    B.name=fname.replace(".gml","_gml")
    if fnamei:
        B.namei=fnamei[:-4]
    B.ego=ego
    B.friendship=bool(fname)
    B.interaction=bool(fnamei)
    B.sid=sid
    B.uid=uid
    B.scriptpath=scriptpath
    B.fb_link=fb_link
    B.dpath=dpath
    B.fpath=fpath
    B.prefix="https://raw.githubusercontent.com/OpenLinkedSocialData/{}master/".format(umbrella_dir)
    B.umbrella_dir=umbrella_dir
    c("antes de ler")
    #fnet=S.fb.readGML(dpath+fname)     # return networkx graph
    fnet=S.fb.readGML2(dpath+fname)     # return networkx graph
#    return fnet
    c("depois de ler, antes de fazer rdf")
    fnet_=rdfFriendshipNetwork(fnet)   # return rdflib graph
    if B.interaction:
        inet=S.fb.readGML(dpath+fnamei)    # return networkx graph
        inet_=rdfInteractionNetwork(inet)      # return rdflib graph
    else:
        inet_=0
    meta=makeMetadata(fnet_,inet_)     # return rdflib graph with metadata about the structure
    c("depois de rdf, escrita em disco")
    writeAllFB(fnet_,inet_,meta)  # write linked data tree
    c("cabo")