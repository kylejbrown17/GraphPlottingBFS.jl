module GraphPlottingBFS

using LightGraphs, MetaGraphs
using Compose
using Colors
using Parameters

export
	BFS_state,
	bfs!,
	plot_graph_bfs

@with_kw struct BFS_state
    d::Int     = 0 # current depth
	w::Int     = 0 # current width
	d_max::Int = 0 # max depth so far
end


function leaf_case!(G,v,s)
    set_prop!(G,v,:height,0)   # distance to leaf
    set_prop!(G,v,:depth, s.d) # distance to root
    set_prop!(G,v,:left,  s.w)
    set_prop!(G,v,:width, 1)
    set_prop!(G,v,:right, s.w+1)
    set_prop!(G,v,:center, (get_prop(G,v,:left)+get_prop(G,v,:right))/2.0)
    s = BFS_state(d=s.d,w=s.w+1,d_max=max(s.d_max,s.d))
end
function initial_branch_case!(G,v,s)
    set_prop!(G,v,:height,0)   # distance to leaf
    set_prop!(G,v,:left,  s.w)
    set_prop!(G,v,:depth,  s.d)
    set_prop!(G,v,:edge_count,  0)
    set_prop!(G,v,:center, 0.0)
end
function backup!(G,v,v2,s)
    set_prop!(G,v,:height, max(get_prop(G,v,:height),1+get_prop(G,v2,:height)))   # distance to leaf
    set_prop!(G,v,:right, s.w)
    set_prop!(G,v,:width, s.w - get_prop(G,v,:left))
    set_prop!(G,v,:edge_count,  1 + get_prop(G,v,:edge_count))
    c = get_prop(G,v,:center)
    e = get_prop(G,v,:edge_count)
    set_prop!(G,v,:center, c*((e-1)/e) + get_prop(G,v2,:center)*(1/e))
end


function bfs!(G,v,s)
    s = BFS_state(d=s.d+1,w=s.w,d_max=s.d_max)
    if length(inneighbors(G,v)) == 0
        s = leaf_case!(G,v,s)
    else
        initial_branch_case!(G,v,s)
        for v2 in inneighbors(G,v)
            s = bfs!(G,v2,s)
            backup!(G,v,v2,s)
        end
    end
    return BFS_state(d=s.d-1,w=s.w,d_max=s.d_max)
end

function plot_graph_bfs(graph,v=0;mode=:leaf_aligned,ϵ=0.000000001,edge_pad=1.1,fillcolor="cyan")
    G = MetaDiGraph(graph)
    if v == 0
        end_vtxs = [v for v in vertices(G) if length(outneighbors(G,v)) == 0]
        s = BFS_state(0,0,0)
        for v in end_vtxs
            s = bfs!(G,v,s)
        end
    else
        s = BFS_state(0,0,0)
        s = bfs!(G,v,s)
    end
    set_default_graphic_size((s.d_max*2)cm,(s.w)cm)
    if mode == :leaf_aligned
        x = [get_prop(G,v,:height)+0.5 for v in vertices(G)]
    else
        x = s.d_max .- [get_prop(G,v,:depth)-0.5 for v in vertices(G)]
    end
    y = [get_prop(G,v,:center) for v in vertices(G)]
    r = 0.2;
    rp = edge_pad*r; # padded radius for plotting
    lines = Vector{Compose.Form}()
    for e in edges(G)
        dx = (x[e.dst] - x[e.src])
        dy = (y[e.dst] - y[e.src])
        d = sqrt(dx^2 + dy^2)
        push!(lines,
            line([
                (x[e.src] + (rp)*dx/d, y[e.src] + (rp)*dy/d),
                (x[e.dst] - (rp)*dx/d, y[e.dst] - (rp)*dy/d)
                ])
        )
    end
    compose(context(units=UnitBox(0,0,s.d_max,s.w)),
        (context(),
            [text(x[i],y[i],string(i),hcenter,vcenter) for i in 1:nv(G)]...,
            stroke("black"),fontsize(10pt), font("futura")),
        (context(),circle(x,y,r*ones(nv(G))),fill(fillcolor)),
        (context(),lines...,stroke("black"))
    )
end

end # module
