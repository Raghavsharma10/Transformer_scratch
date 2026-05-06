async def main():
    """
    Main code (asynchronous requests)

    You can send one millions request with aiohttp :

    https://pawelmhm.github.io/asyncio/python/aiohttp/2016/04/22/asyncio-aiohttp.html

    But don't do that on one server, it's DDOS !

    """
    # Create Client from endpoint string in Duniter format
    client = Client(BMAS_ENDPOINT)
    tasks = []

    # Get the node summary infos by dedicated method (with json schema validation)
    print("\nCall bma.node.summary:")
    task = asyncio.ensure_future(client(bma.node.summary))
    tasks.append(task)

    # Get the money parameters located in the first block
    print("\nCall bma.blockchain.parameters:")
    task = asyncio.ensure_future(client(bma.blockchain.parameters))
    tasks.append(task)

    responses = await asyncio.gather(*tasks)
    # you now have all response bodies in this variable
    print("\nResponses:")
    print(responses)

    # Close client aiohttp session
    await client.close()