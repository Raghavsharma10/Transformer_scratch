def makeRetweetNetwork(tweets):
    """Receives tweets, returns directed retweet networks.
    
    Without and with isolated nodes.
    """
    G=x.DiGraph()
    G_=x.DiGraph()
    for tweet in tweets:
        text=tweet["text"]
        us=tweet["user"]["screen_name"]
        if text.startswith("RT @"):
            prev_us=text.split(":")[0].split("@")[1]
            #print(us,prev_us,text)
            if G.has_edge(prev_us,us):
                G[prev_us][us]["weight"]+=1
                G_[prev_us][us]["weight"]+=1
            else:
                G.add_edge(prev_us, us, weight=1.)
                G_.add_edge(prev_us, us, weight=1.)
        if us not in G_.nodes():
            G_.add_node(us)
    return G,G_