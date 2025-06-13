using cuTENSOR
import TensorOperations: CUDAAllocator, cuTENSORBackend
const GPU_ALLOCATOR = TensorOperations.CUDAAllocator{CUDA.default_memory, CUDA.default_memory, CUDA.default_memory}()

drop=Iterators.drop

#returns list of the orthoganality centres of A, assumes A is right-normalised
function orthcentersmps(A::TreeNetwork)
    B = deepcopy(A.sites)
    for (id, X) in drop(Traverse(A), 1)
        nc = length(A.tree[id].children)
        par = A.tree[id].parent
        dir = findbond(A.tree[par], id)
        AL, C = QR(B[par], dir)
        B[id] = contractC!(B[id], C, 1)
    end
    return B
end

"""
    physdims(M::TreeNetwork)

Return the physical dimensions of a tree-MPS or tree-MPO `M`.
"""
function physdims(M::TreeNetwork)
    N = length(M)
    res = Vector{Int}(undef, N)
    for (i, site) in enumerate(M)
        res[i] = size(site)[end]
    end
    return Dims(res)
end

function mpsrightnorm!(net::TreeNetwork, id::Int)
    loopcheck(net)
    children = net.tree[id].children
    nc = length(children)
    for (i, child) in enumerate(children)
        length(net.tree[child].children) >= 1 && mpsrightnorm!(net, child)
        dchild = size(net[child])
        dpar = size(net[id])
        
        reshaped_net_child_for_lq = reshape(net[child], dchild[1], :)
        input_to_lq = CuArray(reshaped_net_child_for_lq)
        C, AR_from_lq = lq(input_to_lq)

        temp_version_of_AR = CuArray(AR_from_lq) # Matrix(AR_from_lq)
        
        final_net_child_val = reshape(temp_version_of_AR, dchild)
        
        net[child] = final_net_child_val

        Cunetid = CuArray(net[id])
        
        IC=collect(1:nc+2)
        IA=collect(1:nc+2)
        IC[i+1]=-1
        net[id] = tensorcontract(IC, Cunetid, IA, C, [i+1,-1]; backend=cuTENSORBackend(), allocator=GPU_ALLOCATOR)
        C = nothing
    end
end

"""
    mpsrightnorm!(A::TreeNetwork)

When applied to a tree-MPS, right normalise towards head-node.

"""
mpsrightnorm!(net::TreeNetwork) = mpsrightnorm!(net, findheadnode(net))

"""
    mpsmixednorm!(A::TreeNetwork, id::Int)

Normalise tree-MPS `A` such that orthogonality centre is on site `id`.

"""
function mpsmixednorm!(net::TreeNetwork, id::Int)
    1 <= id <= length(net) || throw(BoundsError(net.tree, id))
    setheadnode!(net, id)
    mpsrightnorm!(net, id)
end

"""
    mpsmoveoc!(A::TreeNetwork, id::Int)

Move the orthogonality centre of right normalised tree-MPS `A` to site `id`.

This function will be more efficient than using `mpsmixednorm!` if the tree-MPS is already right-normalised.

"""
function mpsmoveoc!(A::TreeNetwork, id::Int)
    for site in drop(pathfromhead(A.tree, id), 1)
        mpsshiftoc!(A, site)
    end
end

"""
    mpsshiftoc!(A::TreeNetwork, newhd::Int)

Shift the orthogonality centre by one site, setting new head-node `newhd`.

"""
function mpsshiftoc!(A::TreeNetwork, newhd::Int)
    oldhd = findheadnode(A)
    in(newhd, A.tree[oldhd].children) || throw("site $id is not child of head-node")

    setheadnode!(A, newhd)
    AL, C = QR(A[oldhd], 1)
    A[oldhd] = AL
    A[newhd] = contractC!(A[newhd], C, findchild(A.tree[newhd], oldhd)+1)
end

function calcbonddims!(tree::Tree, physdims::Dims, Dmax::Int, M::Array{Int, 2}, id::Int)
    for child in tree[id].children
        length(tree[child].children) >= 1 && calcbonddims!(tree, physdims, Dmax, M, child)
        D = physdims[child]
        for grandchild in tree[child].children
            D *= M[child, grandchild]
        end
        M[id, child] = min(D, Dmax)
        M[child, id] = min(D, Dmax)
    end
end
function calcbonddims(tree::Tree, physdims::Dims, Dmax::Int)
    loopcheck(tree)
    N = length(tree)
    M = zeros(Int, N, N)
    calcbonddims!(tree, physdims, Dmax, M, findheadnode(tree))
    return M
end

function normmps(net::TreeNetwork, id::Int)
    nc = length(net.tree[id].children)
    IA = collect(1:nc+2)
    IB = collect(nc+3:2*nc+4)
    IA[end] = -1 #contract physical indices
    IB[end] = -1 #contract physical indices 
    ρ = tensorcontract(net[id], IA, conj(net[id]), IB; backend=cuTENSORBackend(), allocator=GPU_ALLOCATOR)
    for (i, child) in enumerate(net.tree[id].children)
        ρchild = normmps(net, child)
        nd = (nc+2-i)*2
        halfnd = div(nd,2)
        IA = collect(1:nd)
        IA[2] = -1
        IA[halfnd+2] = -2
        ρ = tensorcontract(ρ, IA, ρchild, [-1, -2]; backend=cuTENSORBackend(), allocator=GPU_ALLOCATOR)
    end
    return ρ
end

"""
    normmps(net::TreeNetwork; mpsorthog=:None)
    
When applied to a tree-MPS `mpsorthog=:Left` is not defined.

"""
function normmps(net::TreeNetwork; mpsorthog=:None)
    loopcheck(net)
    if mpsorthog==:Right
        OC = findheadnode(net)
        AC = net[OC]
        nd = ndims(AC)
        IA = collect(1:nd)
        return real(scalar(tensorcontract(AC, IA, conj(AC), IA; backend=cuTENSORBackend(), allocator=GPU_ALLOCATOR)))
    elseif typeof(mpsorthog)<:Int
        OC = mpsorthog
        AC = net[OC]
        nd = ndims(AC)
        IA = collect(1:nd)
        return real(scalar(tensorcontract(AC, IA, conj(AC), IA; backend=cuTENSORBackend(), allocator=GPU_ALLOCATOR)))
    elseif mpsorthog==:None
        hn = findheadnode(net)
        ρ = normmps(net, hn)
        return real(ρ[1])
    end
end

function initenvs!(A::TreeNetwork, M::TreeNetwork, F::AbstractVector, id::Int)
    for child in A.tree[id].children
        F = initenvs!(A, M, F, child)
    end
    F[id] = updaterightenv(A[id], M[id], F[A.tree[id].children]...)
    return F
end
function initenvs(A::TreeNetwork, M::TreeNetwork, F::Nothing)
    hn = findheadnode(A)
    N = length(A)
    F = Vector{Any}(undef, N)
    for child in A.tree[hn].children
        F = initenvs!(A, M, F, child)
    end
    return F
end
function initenvs(A::TreeNetwork, M::TreeNetwork, F::AbstractVector)
    return F
end

tdvp1sweep!(dt, A::TreeNetwork, M::TreeNetwork, F=nothing, timestep_idx::Int=0; verbose=false, kwargs...) =
    tdvp1sweep!(dt, A, M, initenvs(A, M, F), findheadnode(A), timestep_idx; verbose=verbose, kwargs...)

"""
    tdvp1sweep!(dt, A::TreeNetwork, M::TreeNetwork, F::AbstractVector, id::Int, timestep_idx::Int; verbose=false, kwargs...)

Propagates the tree-MPS A with the tree-MPO M following the 1-site TDVP method. The sweep is done back and forth with a time step dt/2. F represents the merged left and right parts of the site being propagated.  
"""
function tdvp1sweep!(dt, A::TreeNetwork, M::TreeNetwork, F::AbstractVector, id::Int, timestep_idx::Int; verbose=false, kwargs...)

    children = A.tree[id].children
    parent = A.tree[id].parent
    AC = A[id]

    F0 = parent==0 ? fill!(similar(M[1], (1,1,1)), 1) : F[parent]
    
    #(OC begins on node)
    #evolve node forward half a time step
    exponentiate!(x->applyH1(x, M[id], F0, F[children]...), -im*dt/2, AC; ishermitian=true)

    for (i, child) in enumerate(children)
        
        grandchildren = A.tree[child].children
        otherchildren = filter(x->x!=child, children)
        
        AL, C = QR(AC, i+1)

        F[id] = updateleftenv(AL, M[id], i, F0, F[otherchildren]...)

        exponentiate!(x->applyH0(x, F[id], F[child]), im*dt/2, C; ishermitian=true)

        A[child] = contractC!(A[child], C, 1)
        
        A, F = tdvp1sweep!(dt, A, M, F, child, timestep_idx; verbose=verbose, kwargs...)
        
        AR, C = QR(A[child], 1)
        
        A[child] = AR
        
        F[child] = updaterightenv(AR, M[child], F[grandchildren]...)

        exponentiate!(x->applyH0(x, F[child], F[id]), im*dt/2, C; ishermitian=true)
        
        AC = contractC!(AL, C, i+1)
        C = nothing
    end

    #evolve node forward half a time step
    exponentiate!(x->applyH1(x, M[id], F0, F[children]...), -im*dt/2, AC; ishermitian=true)

    A[id] = AC

    return A, F
end

"""
    productstatemps(tree_::Tree, physdims::Dims, Dmax::Int=1; state=:Vacuum)

Return a tree-MPS representing a product state with local Hilbert space dimensions given by `physdims`.

By default all bond-dimensions will be 1 since the state is a product state. However, to
embed the product state in a manifold of greater bond-dimension, `Dmax` can be set accordingly.

The indvidual states of the MPS sites can be provided by setting `state` to a list of
column vectors. Setting `state=:Vacuum` will produce an MPS in the vacuum state (where the
state of each site is represented by a column vector with a 1 in the first row and zeros
elsewhere). Setting `state=:FullOccupy` will produce an MPS in which each site is fully
occupied (ie. a column vector with a 1 in the last row and zeros elsewhere).

# Example

```julia-repl
julia> ψ = unitcol(1,2); d = 6; N = 30; α = 0.1; Δ = 0.0; ω0 = 0.2; s = 1

julia> cpars = chaincoeffs_ohmic(N, α, s)

julia> H = spinbosonmpo(ω0, Δ, d, N, cpars, tree=true)

julia> A = productstatemps(H.tree, physdims(H), state=[ψ, fill(unitcol(1,d), N)...]) # tree-MPS representation of |ψ>|Vacuum>
```
"""
function productstatemps(tree_::Tree, physdims::Dims, Dmax::Int=1; state=:Vacuum)
    tree = deepcopy(tree_)
    hn = findheadnode(tree)
    leafnodes = leaves(tree)
    N = length(tree)
    setheadnode!(tree, leafnodes[1])
    bonddims1 = calcbonddims(tree, physdims, Dmax)
    setheadnode!(tree, leafnodes[2])
    bonddims2 = calcbonddims(tree, physdims, Dmax)
    bonddims = min.(bonddims1, bonddims2)
    tree = deepcopy(tree_)
    
    if state==:Vacuum
        statelist = [unitcol(1, physdims[i]) for i in 1:N]
    elseif state==:FullOccupy
        statelist = [unitcol(physdims[i], physdims[i]) for i in 1:N]
    elseif typeof(state)<:Vector
        statelist = state
    else
        throw(ErrorException("state input not recognised"))
    end

    A = Vector{Any}(undef, N)

    for (id, node) in enumerate(tree)
        Dpar = id==hn ? 1 : bonddims[id, node.parent]
        Dchildren = bonddims[id, node.children]
        B_flat = CUDA.zeros(ComplexF64, Dpar, prod(Dchildren), physdims[id])
        for j in 1:min(Dpar, prod(Dchildren))
            if physdims[id] > 0
                cpu_slice = zeros(ComplexF64, physdims[id])
                cpu_slice[:] = statelist[id]
                copyto!(view(B_flat, j, j, 1:physdims[id]), statelist[id])
            end
        end
        B = reshape(B_flat, Dpar, Dchildren..., physdims[id])
        A[id] = B
    end
    net = TreeNetwork(tree, A)

    mpsrightnorm!(net)
    return net
end
productstatemps(tree::Tree, physdims::Int, Dmax::Int; state=:Vacuum) =
    productstatemps(tree, ntuple(i -> physdims, length(tree)), Dmax; state=state)

"""
    mpsembed(A::TreeNetwork, Dmax::Int)

Embed tree-MPS `A` in manifold of max bond-dimension `Dmax`.

"""
function mpsembed!(A::TreeNetwork, Dmax::Int)
    tree = deepcopy(A.tree)
    pdims = physdims(A)
    hn = findheadnode(tree)
    leafnodes = leaves(tree)
    setheadnode!(tree, leafnodes[1])
    bonddims1 = calcbonddims(tree, pdims, Dmax)
    setheadnode!(tree, leafnodes[2])
    bonddims2 = calcbonddims(tree, pdims, Dmax)
    bonddims = min.(bonddims1, bonddims2)

    for (id, nd) in enumerate(A.tree.nodes)
        parent = nd.parent
        children = nd.children

        if parent != 0
            A[id] = setbond(A[id], bonddims[id, [parent, children...]]...)
        else
            A[id] = setbond(A[id], 1, bonddims[id, children]...)
        end

    end

    return A
end

"""
    to_gpu(net::TreeNetwork)

Convert a TreeNetwork's site tensors to CuArrays if they are not already.
"""
function to_gpu(net::TreeNetwork)
    gpu_sites = Vector{Any}(undef, length(net.sites))
    for i in 1:length(net.sites)
        if net.sites[i] isa AbstractArray && !(net.sites[i] isa CuArray)
            gpu_sites[i] = CuArray(net.sites[i])
        elseif net.sites[i] isa CuArray
            gpu_sites[i] = net.sites[i]
        else
            gpu_sites[i] = net.sites[i]
        end
    end
    return TreeNetwork(net.tree, gpu_sites)
end

"""
    bonddims(A::TreeNetwork)

Return the bon-dimension of a tree-MPS `A`.

"""
function bonddims(A::TreeNetwork)
    N = length(A)
    mat = zeros(Int, N, N)
    for bond in bonds(A)
        id1 = bond[1]
        id2 = bond[2]
        dir = findbond(A.tree[id1], id2)
        D = size(A[id1])[dir]
        mat[bond...] = D
        mat[reverse(bond)...] = D
    end
    mat
end
