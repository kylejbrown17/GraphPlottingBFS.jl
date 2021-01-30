using GraphPlottingBFS
import Cairo
using Compose

g = DiGraph(12)

add_edge!(g,1,2)
add_edge!(g,1,3)
add_edge!(g,2,4)
add_edge!(g,3,4)
add_edge!(g,4,5)
add_edge!(g,4,6)

add_edge!(g,7,8)
add_edge!(g,7,9)
add_edge!(g,8,10)
add_edge!(g,9,10)
add_edge!(g,10,5)
add_edge!(g,10,11)
add_edge!(g,10,12)

display_graph(g,align_mode=:root_aligned)


feats = [
    GraphPlottingBFS.ForwardDepth(),
    GraphPlottingBFS.ForwardWidth(),
    GraphPlottingBFS.ForwardTreeWidth(),
    GraphPlottingBFS.BackwardDepth(),
    GraphPlottingBFS.BackwardWidth(),
    GraphPlottingBFS.BackwardTreeWidth(),
]

feat_vals = GraphPlottingBFS.forward_pass!(g,feats)
feat_vals = GraphPlottingBFS.forward_pass!(g,feats,feat_vals)