import networkx as nx 

def pagerank(G, alpha=0.85, personalization=None, 
			max_iter=100, tol=1.0e-6, nstart=None, weight='weight', 
			dangling=None): 
	if len(G) == 0: 
		return {} 

	if not G.is_directed(): 
		D = G.to_directed() 
	else: 
		D = G 

	# Create a copy in (right) stochastic form 
	W = nx.stochastic_graph(D, weight=weight) 
	N = W.number_of_nodes() 

	# Choose fixed starting vector if not given 
	if nstart is None: 
		x = dict.fromkeys(W, 1.0 / N) 
	else: 
		# Normalized nstart vector 
		s = float(sum(nstart.values())) 
		x = dict((k, v / s) for k, v in nstart.items()) 

	if personalization is None: 

		# Assign uniform personalization vector if not given 
		p = dict.fromkeys(W, 1.0 / N) 
	else: 
		missing = set(G) - set(personalization) 
		if missing: 
			raise NetworkXError('Personalization dictionary '
								'must have a value for every node. '
								'Missing nodes %s' % missing) 
		s = float(sum(personalization.values())) 
		p = dict((k, v / s) for k, v in personalization.items()) 

	if dangling is None: 

		# Use personalization vector if dangling vector not specified 
		dangling_weights = p 
	else: 
		missing = set(G) - set(dangling) 
		if missing: 
			raise NetworkXError('Dangling node dictionary '
								'must have a value for every node. '
								'Missing nodes %s' % missing) 
		s = float(sum(dangling.values())) 
		dangling_weights = dict((k, v/s) for k, v in dangling.items()) 
	dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0] 

	# power iteration: make up to max_iter iterations 
	for _ in range(max_iter): 
		xlast = x 
		x = dict.fromkeys(xlast.keys(), 0) 
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes) 
		for n in x: 

			# this matrix multiply looks odd because it is 
			# doing a left multiply x^T=xlast^T*W 
			for nbr in W[n]: 
				x[nbr] += alpha * xlast[n] * W[n][nbr][weight] 
			x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] 

		# check convergence, l1 norm 
		err = sum([abs(x[n] - xlast[n]) for n in x]) 
		if err < N*tol: 
			return x 
	raise NetworkXError('pagerank: power iteration failed to converge '
						'in %d iterations.' % max_iter) 


print("### Welcome to a PageRank demonstration ###")
print("( the algorithm used can be found at: https://networkx.github.io/documentation/networkx-1.9/index.html )")
ans = True
while ans:
	print ("""
	Select one:
	1.Learn more about the algorithm
	2.Test the algorithm
	3.Exit/Quit
	""")
	ans = input("What would you like to do? ") 
	if ans == "1": 
		print("\nPageRank is a link analysis algorithm and it assigns a numerical weighting to each element of a hyperlinked set of documents, such as the World Wide Web, with the purpose of 'measuring' its relative importance within the set. \nThe algorithm may be applied to any collection of entities with reciprocal quotations and references.\nThe numerical weight that it assigns to any given element E is referred to as the PageRank of E") 
	
	elif ans == "2":
		print("\n The PageRank function takes in several arguments, lets check what they are")
		print("\n pagerank(Graph, Alpha, Personalization, Maximum iteration, Error tolerance, Starting value for each node, Weight, Dangling)")
		print("\n -> Graph: A NetworkX graph. Undirected graphs will be converted to a directed graph with two directed edges for each undirected edge. ")
		print("\n -> Alpha (float, optional) : Damping parameter for PageRank, default=0.85.")
		print("\n -> Personalization (dict, optional) : The 'personalization vector' consisting of a dictionary with a key for every graph node and nonzero personalization value for each node. By default, a uniform distribution is used. ")
		print("\n -> Maximum iteration (integer, optional) : Maximum number of iterations in power method eigenvalue solver. ")
		print("\n -> Error tolerance (float, optional) : Error tolerance used to check convergence in power method solver.")
		print("\n -> Starting value for each node (dictionary, optional ): Starting value of PageRank iteration for each node. ")
		print("\n -> Weight (key, optional) : Edge data key to use as weight. If None weights are set to 1. ")
		print("\n -> Dangling (dict, optional) : The outedges to be assigned to any 'dangling' nodes, i.e., nodes without any outedges. The dict key is the node the outedge points to and the dict value is the weight of that outedge. By default, dangling nodes are given outedges according to the personalization vector (uniform if not specified). This must be selected to result in an irreducible transition matrix (see notes under google_matrix). It may be common to have the dangling dict to be the same as the personalization dict. ")

		print("\n Graph (Attention: must have m >= 1 and m < n)")
		graph_n = int(input("n value: "))
		print(type(graph_n))
		graph_m = int(input("m value: "))
		print(type(graph_m))

		alpha = int(input("Optional, default value = 0.85, type '1' for default value: "))
		if alpha == 1:
			alpha = 0.85

		personalization = input("Personalization, type = dict, type 'yes' for default value: ")
		if personalization.upper() == 'YES':
			personalization = None

		max_it = int(input("Maximum Iteration, type '1' for the default(100) value: "))
		if max_it == 1:
			max_it = 100

		tol = int(input("Error tolerance, type '1' for deault(1.0e-6) value: "))
		if tol == 1:
			tol = 1.0e-6

		sval = input("Starting value for each node (dictionary type), Optional, type 'yes' for no value: ")
		if sval.upper() == 'YES':
			sval = None

		w = input("Weight, key, optional, type 'yes' for the default value: ")
		if w.upper() == 'YES':
			w = 'weight'

		dang = input("Dangling (dict, optional), type 'yes' for the default value: ")
		if dang.upper() == 'YES':
			dang = None

								
		graph = nx.barabasi_albert_graph(graph_n,graph_m)
		print("\n Graph")
		for i in graph:
			print (i)

		r = pagerank(graph, alpha, personalization, max_it, tol, sval, w, dang)
		print("\n PageRank values")
		print(r) 


	elif ans == "3":
		print("\n Goodbye")
		break
	
	elif ans != "":
		print("\n Not Valid Choice Try again") 