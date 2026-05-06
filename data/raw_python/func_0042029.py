async def main():
    """
    Main code (synchronous requests)
    """
    # Create Client from endpoint string in Duniter format
    client = Client(BMAS_ENDPOINT)

    # Get the node summary infos by dedicated method (with json schema validation)
    print("\nCall bma.node.summary:")
    response = await client(bma.node.summary)
    print(response)

    # Get the money parameters located in the first block
    print("\nCall bma.blockchain.parameters:")
    response = await client(bma.blockchain.parameters)
    print(response)

    # Get the current block
    print("\nCall bma.blockchain.current:")
    response = await client(bma.blockchain.current)
    print(response)

    # Get the block number 10
    print("\nCall bma.blockchain.block(10):")
    response = await client(bma.blockchain.block, 10)
    print(response)

    # jsonschema validator
    summary_schema = {
        "type": "object",
        "properties": {
            "duniter": {
                "type": "object",
                "properties": {
                    "software": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string",
                    },
                    "forkWindowSize": {
                        "type": "number"
                    }
                },
                "required": ["software", "version"]
            },
        },
        "required": ["duniter"]
    }

    # Get the node summary infos (direct REST GET request)
    print("\nCall direct get on node/summary")
    response = await client.get('node/summary', rtype=RESPONSE_AIOHTTP, schema=summary_schema)
    print(response)

    # Close client aiohttp session
    await client.close()