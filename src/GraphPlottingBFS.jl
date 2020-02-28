module GraphPlottingBFS

using LightGraphs, MetaGraphs
using Compose
using Colors
using Parameters

export
	BFS_state,
    leaf_case!,
    initial_branch_case!,
    backup!,
    step_in,
    step_out,
	bfs!,
	get_graph_bfs,
	display_graph_bfs,
	plot_graph_bfs,
    render_grid_graph

@with_kw struct BFS_state
    d::Int     = 0 # current depth
	w::Int     = 0 # current width
	d_max::Int = 0 # max depth so far
end


function leaf_case!(G,v,s::BFS_state)
    set_prop!(G,v,:height, 0)   # distance to leaf
    set_prop!(G,v,:depth, s.d) # distance to root
    set_prop!(G,v,:left,  s.w)
    set_prop!(G,v,:width, 1)
    set_prop!(G,v,:right, s.w+1)
    set_prop!(G,v,:center, (get_prop(G,v,:left)+get_prop(G,v,:right))/2.0)
    s = BFS_state(d=s.d,w=s.w+1,d_max=max(s.d_max,s.d))
end
function initial_branch_case!(G,v,s::BFS_state)
    set_prop!(G,v,:height, 0)   # distance to leaf
    set_prop!(G,v,:left,  s.w)
    set_prop!(G,v,:depth,  s.d)
    set_prop!(G,v,:edge_count,  0)
    set_prop!(G,v,:center, 0.0)
end
function backup!(G,v,v2,s::BFS_state)
    set_prop!(G,v,:height, max(get_prop(G,v,:height),1+get_prop(G,v2,:height)))   # distance to leaf
    set_prop!(G,v,:right, s.w)
    set_prop!(G,v,:width, s.w - get_prop(G,v,:left))
    set_prop!(G,v,:edge_count,  1 + get_prop(G,v,:edge_count))
    c = get_prop(G,v,:center)
    e = get_prop(G,v,:edge_count)
    set_prop!(G,v,:center, c*((e-1)/e) + get_prop(G,v2,:center)*(1/e))
end

step_in(s::BFS_state) = BFS_state(d=s.d+1,w=s.w,d_max=s.d_max)
step_out(s::BFS_state) = BFS_state(d=s.d-1,w=s.w,d_max=s.d_max)

function bfs!(G,v,s)
    s = step_in(s)
    if indegree(G,v) == 0
        s = leaf_case!(G,v,s)
    else
        initial_branch_case!(G,v,s)
        for v2 in inneighbors(G,v)
            s = bfs!(G,v2,s)
            backup!(G,v,v2,s)
        end
    end
    return step_out(s)
end

abstract type GraphInfoFeature end
struct ForwardDepth <: GraphInfoFeature end
struct BackwardDepth <: GraphInfoFeature end
struct ForwardWidth <: GraphInfoFeature end
struct BackwardWidth <: GraphInfoFeature end
struct VtxWidth <: GraphInfoFeature end
struct ForwardIndex <: GraphInfoFeature end
struct ForwardCenter <: GraphInfoFeature end
struct BackwardIndex <: GraphInfoFeature end
struct BackwardCenter <: GraphInfoFeature end

initial_value(f) = 1.0
initial_value(f::Union{ForwardDepth,BackwardDepth}) = 1
forward_propagate(f,G,v,val,v2,val2,args...) = val
backward_propagate(f,G,v,val,v2,val2,args...) = val
forward_accumulate(f,G,v,val,vtxs,vals,args...) = val
backward_accumulate(f,G,v,val,vtxs,vals,args...) = val

# forward_propagate(f::BackwardDepth,	G,v,val,v2,val2,args...) 	= max(val,val2-1)
backward_propagate(f::BackwardDepth,G,v,val,v2,val2,args...) 	= max(val,val2+1)
# backward_propagate(f::ForwardDepth,G,v,val,v2,val2,args...) 	= max(val,val2-1)
forward_propagate(f::ForwardDepth,	G,v,val,v2,val2,args...) 	= max(val,val2+1)
forward_propagate(f::ForwardWidth,	G,v,val,v2,val2,args...) 	= max(val,val2/max(1,outdegree(G,v2)))
backward_propagate(f::BackwardWidth,G,v,val,v2,val2,args...) 	= max(val,val2/max(1,indegree(G,v2)))
forward_accumulate(f::ForwardWidth,	G,v,val,vtxs,vals) 			= max(val,sum([0,map(i->vals[i]/outdegree(G,vtxs[i]),1:length(vals))...]))
backward_accumulate(f::BackwardWidth,G,v,val,vtxs,vals) 		= max(val,sum([0,map(i->vals[i]/indegree(G,vtxs[i]),1:length(vals))...]))

function get_graph_layout(G,feats=[ForwardDepth(),BackwardDepth(),ForwardWidth(),BackwardWidth()])
	@assert !is_cyclic(G)
	feat_vals = Dict(f=>map(v->initial_value(f),vertices(G)) for f in feats)
	for v in topological_sort_by_dfs(G)
		for (f,vec) in feat_vals
			vec[v] = forward_accumulate(f,G,v,vec[v],inneighbors(G,v),map(v2->vec[v2], inneighbors(G,v)))
			for v2 in inneighbors(G,v)
				vec[v] = forward_propagate(f,G,v,vec[v],v2,vec[v2])
			end
		end
	end
	for v in topological_sort_by_dfs(reverse(G))
		for (f,vec) in feat_vals
			vec[v] = backward_accumulate(f,G,v,vec[v],outneighbors(G,v),map(v2->vec[v2], outneighbors(G,v)))
			for v2 in outneighbors(G,v)
				vec[v] = backward_propagate(f,G,v,vec[v],v2,vec[v2])
			end
		end
	end
	feat_vals[VtxWidth()] = max.(feat_vals[ForwardWidth()],feat_vals[BackwardWidth()])

	graph = MetaDiGraph(deepcopy(G))
	end_vtxs = [v for v in vertices(graph) if outdegree(graph,v) == 0]
	s = BFS_state(0,0,0)
	for v in end_vtxs
		s = bfs!(graph,v,s)
	end
	for (idx_key,ctr_key,depth_key) in [(ForwardIndex(),ForwardCenter(),ForwardDepth()),((BackwardIndex(),BackwardCenter(),BackwardDepth()))]
		vec = feat_vals[depth_key]
		forward_width_counts = map(v->Int[],1:maximum(vec))
		forward_idxs = zeros(nv(G))
		backward_idxs = zeros(nv(G))
		for v in topological_sort_by_dfs(G)
			push!(forward_width_counts[vec[v]],v)
		end
		for vec in forward_width_counts
			sort!(vec,by=v->get_prop(graph,v,:left))
			for (i,v) in enumerate(vec)
				if i > 1
					v2 = vec[i-1]
					forward_idxs[v] += forward_idxs[v2] + feat_vals[VtxWidth()][v2]
				end
				if indegree(G,v) > 0
					min_idx = minimum(map(v2->forward_idxs[v2],inneighbors(G,v)))
					forward_idxs[v] = max(forward_idxs[v],min_idx)
					for v2 in vec[1:i-1]
						forward_idxs[v] = max(forward_idxs[v],forward_idxs[v2]+feat_vals[VtxWidth()][v2])
					end
				end
			end
		end
		feat_vals[idx_key] = forward_idxs
		feat_vals[ctr_key] = forward_idxs .+ 0.5*feat_vals[VtxWidth()]
	end

	feat_vals
end


function get_graph_bfs(graph,v=0;
        shape_function = (G,v,x,y,r)->Compose.circle(x,y,r),
        color_function = (G,v,x,y,r)->"cyan",
        text_function = (G,v,x,y,r)->string(v),
        mode=:leaf_aligned,
        ϵ=0.000000001,
        edge_pad=1.1,
		aspect_ratio=2,
        r = 0.2)

    # G = graph
	feat_vals = get_graph_layout(graph)
    if mode == :leaf_aligned
        x = feat_vals[ForwardDepth()]
	    y = feat_vals[ForwardCenter()]
    else
        x = feat_vals[BackwardDepth()]
		x = 1 + maximum(x) .- x
	    y = feat_vals[BackwardCenter()]
    end
	x = x .+ (1 - minimum(x))
	y = y .+ (1 - minimum(y))
	# canvas_height = maximum(feat_vals[ForwardDepth()])+1
	# canvas_width = maximum(feat_vals[ForwardWidth()])+1
    # set_default_graphic_size((2*canvas_height)cm,(canvas_width)cm)
	canvas_height = 1+maximum(x) .- min(0,minimum(x))
	canvas_width = 1+maximum(y) .- min(0,minimum(y))
    set_default_graphic_size((aspect_ratio*canvas_height)cm,(canvas_width)cm)

    # G = MetaDiGraph(deepcopy(graph))
    # if v == 0
    #     end_vtxs = [v for v in vertices(G) if length(outneighbors(G,v)) == 0]
    #     s = BFS_state(0,0,0)
    #     for v in end_vtxs
    #         s = bfs!(G,v,s)
    #     end
    # else
    #     s = BFS_state(0,0,0)
    #     s = bfs!(G,v,s)
    # end
    # set_default_graphic_size((s.d_max*2)cm,(s.w)cm)
    # if mode == :leaf_aligned
    #     x = [get_prop(G,v,:height)+0.5 for v in vertices(G)]
    # else
    #     x = s.d_max .- [get_prop(G,v,:depth)-0.5 for v in vertices(G)]
    # end
    # y = [get_prop(G,v,:center) for v in vertices(G)]
    rp = edge_pad*r; # padded radius for plotting
    lines = Vector{Compose.Form}()
    for e in edges(graph)
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
    shapes = map(i->shape_function(graph,i,x[i],y[i],r),1:nv(graph))
    fill_colors = map(i->color_function(graph,i,x[i],y[i],r),1:nv(graph))
    text_strings = map(i->text_function(graph,i,x[i],y[i],r),1:nv(graph))

    return canvas_height,canvas_width,x,y,lines,shapes,fill_colors,text_strings
end

function display_graph_bfs(canvas_height,canvas_width,x,y,lines,shapes,fill_colors,text_strings)
    N = length(x)
    compose(context(units=UnitBox(0,0,canvas_height,canvas_width)),
        (context(),[text(x[i],y[i],text_strings[i],hcenter,vcenter) for i in 1:N]...,
            stroke("black"),fontsize(10pt), font("futura")),
        map(i->(context(),shapes[i],fill(fill_colors[i])),1:N)...,
        (context(),lines...,stroke("black")))
end

function plot_graph_bfs(graph,v=0;kwargs...)
    display_graph_bfs(get_graph_bfs(graph,v;kwargs...)...)
end

G = DiGraph(13)
add_edge!(G,1,3)
add_edge!(G,2,3)
add_edge!(G,3,4)

add_edge!(G,5,7)
add_edge!(G,6,7)
add_edge!(G,7,8)

add_edge!(G,4,9)
add_edge!(G,8,9)
add_edge!(G,9,10)
add_edge!(G,10,11)
add_edge!(G,10,12)

add_edge!(G,4,11)
# add_edge!(G,7,4)

plot_graph_bfs(G;mode=:root_aligned)
plot_graph_bfs(G;mode=:leaf_aligned)

"""
    `SimpleGridWorld{G}`

    To simplify rendering.
"""
struct SimpleGridWorld{G}
    graph::G
    x::Vector{Float64}
    y::Vector{Float64}
end
get_x(env::E,i::Int) where {E<:SimpleGridWorld} = env.x[i]
get_y(env::E,i::Int) where {E<:SimpleGridWorld} = env.i[i]

"""
    Tool for rendering paths (sequences of vertices) through a grid environment
"""
function render_grid_graph(G,paths::Vector{Vector{Int}}=Vector{Vector{Int}}();
        width=10,r=0.4,color="gray",pathcolors=["lime","red","blue","violet","orange","cyan"])
    # vertices
    x = Vector{Float64}()
    y = Vector{Float64}()
    for v in vertices(G)
        push!(x,get_prop(G,v,:x))
        push!(y,get_prop(G,v,:y))
    end
    x₀ = minimum(x) - r
    y₀ = minimum(y) - r
    Δx = maximum(x) + r - x₀
    Δy = maximum(y) + r - y₀
    # graphic size
    set_default_graphic_size((width)cm,(width*Δy/Δx)cm)
    # edges
    edge_list = []
    for e in edges(G)
        push!(edge_list, [(x[e.src],y[e.src]),(x[e.dst],y[e.dst])])
    end
    # paths
    rendered_paths = []
    starts = []
    goals = []
    while length(pathcolors) < length(paths)
        pathcolors = [pathcolors..., pathcolors...]
    end
    for (p,color) in zip(paths,pathcolors)
        rpath = []
        start = circle(x[p[1]],y[p[1]],r*0.8)
        goal = circle(x[p[end]],y[p[end]],r*0.8)
        for i in 1:length(p)-1
            push!(rpath, [(x[p[i]],y[p[i]]),(x[p[i+1]],y[p[i+1]])])
        end
        push!(rendered_paths,
            compose(context(),
                (context(),start,fill(color)),
                (context(),goal,fill(color)),
                (context(),line(rpath),stroke(color),linewidth(2pt))
            )
        )
    end

    compose(context(units=UnitBox(x₀, y₀, Δx, Δy)),
        rendered_paths...,
        compose(context(),circle(x,y,r*ones(length(x))),fill(color)),
        compose(context(),line(edge_list),stroke(color),linewidth(4pt))
    )
end

end # module
