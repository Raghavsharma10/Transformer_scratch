def list(self, request):
        """
        Returns the list of documents found on the collection
        """
        pipeline = [{'$match': request.args.pop('match', {})}]

        sort = request.args.pop('sort', {})
        if sort:
            pipeline.append({'$sort': sort})

        project = request.args.pop('project', {})
        if project:
            pipeline.append({'$project': project})

        return Response(serialize(self.collection.aggregate(pipeline)))