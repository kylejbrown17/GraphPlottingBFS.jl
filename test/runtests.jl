using GraphPlottingBFS
using LightGraphs
using Test

let
	N = 4       # num robots
	M = 10      # num delivery tasks
	G = DiGraph(M)
	add_edge!(G,1,4)
	add_edge!(G,2,4)
	add_edge!(G,3,4)
	add_edge!(G,5,7)
	add_edge!(G,6,7)
	add_edge!(G,4,8)
	add_edge!(G,7,8)
	add_edge!(G,8,10)
	add_edge!(G,9,10)
	plot_graph_bfs(G)
end
