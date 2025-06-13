up(ops...) = permutedims(cat(ops...; dims=3), [3,1,2])
dn(ops...) = permutedims(cat(reverse(ops)...; dims=3), [3,1,2])

"""
    hbathchain(N::Int, d::Int, chainparams, longrangecc...; tree=false, reverse=false, coupletox=false)

Generate MPO representing a tight-binding chain of `N` oscillators with `d` Fock states each. Chain parameters are supplied in the standard form: `chainparams` ``=[[ϵ_0,ϵ_1,...],[t_0,t_1,...],c_0]``. The output does not itself represent a complete MPO but will possess an end which is *open* and should be attached to another tensor site, usually representing the *system*.

# Arguments

* `reverse`: If `reverse=true` create a chain were the last (i.e. Nth) site is the site which couples to the system
* `coupletox`: Used to choose the form of the system coupling. `coupletox=true` gives a non-number conserving coupling of the form ``H_{\\text{I}}= A_{\\text{S}}(b_{0}^\\dagger + b_0)`` where ``A_{\\text{S}}`` is a system operator, while `coupletox=false` gives the number-converving coupling ``H_{\\text{I}}=(A_{\\text{S}} b_{0}^\\dagger + A_{\\text{S}}^\\dagger b_0)``
* `tree`: If `true` the resulting chain will be of type `TreeNetwork`; useful for construcing tree-MPOs 

# Example

One can constuct a system site tensor to couple to a chain by using the function `up` to populate the tensor. For example, to construct a system site with Hamiltonian `Hs` and coupling operator `As`, the system tensor `M` is constructed as follows for a non-number conserving interaction:
```julia
u = one(Hs) # system identity
M = zeros(1,3,2,2)
M[1, :, :, :] = up(Hs, As, u)
```

The full MPO can then be constructed with:
```julia
Hmpo = [M, hbathchain(N, d, chainparams, coupletox=true)...]
```

Similarly for a number conserving interaction the site tensor would look like:
```julia
u = one(Hs) # system identity
M = zeros(1,4,2,2)
M[1, :, :, :] = up(Hs, As, As', u)
```
And the full MPO would be
```julia
Hmpo = [M, hbathchain(N, d, chainparams; coupletox=false)...]
```
    
"""
function hbathchain(N::Int, d::Int, chainparams, longrangecc...; tree=false, reverse=false, coupletox=false)
    b = anih(d)
    bd = crea(d)
    n = numb(d)
    u = unitmat(d)
    e = chainparams[1]
    t = chainparams[2]

    numlong = length(longrangecc)
    cc = longrangecc
    D1 = 2 + (coupletox ? 1 : 2)*(1+numlong)
    D2 = coupletox ? D1+1 : D1 

    H=Vector{Any}()

    if coupletox
        M=zeros(D1, D2, d, d)
        M[D1, :, :, :] = up(e[1]*n, t[1]*b, t[1]*bd, fill(zero(u), numlong)..., u)
        M[:, 1, :, :] = dn(e[1]*n, [cc[j][1]*(b+bd) for j=1:numlong]..., b+bd, u)
        for k=1:numlong
            M[k+2,k+3,:,:] = u
        end
        push!(H, M)
    else
        M=zeros(D1, D2, d, d)
        M[D1, :, :, :] = up(e[1]*n, t[1]*b, t[1]*bd, u)
        M[:, 1, :, :] = dn(e[1]*n, b, bd, u)
        numlong > 0 && error("haven't yet coded case of long range couplings with non-hermitian coupling")
        push!(H, M)
    end
    for i=2:N-1
        M=zeros(D2, D2, d, d)
        M[D2, :, :, :] = up(e[i]*n, t[i]*b, t[i]*bd, fill(zero(u), numlong)..., u)
        M[:, 1, :, :] = dn(e[i]*n, [cc[j][i]*(b+bd) for j=1:numlong]..., b, bd, u)
        for k=1:numlong
            M[k+3,k+3,:,:] = u
        end
        push!(H, M)
    end
    M=zeros(D2, d, d)
    M[:, :, :] = dn(e[N]*n, [cc[j][N]*(b+bd) for j=1:numlong]..., b, bd, u)
    push!(H, M)
    if tree
        return TreeNetwork(H)
    else
        H[end] = reshape(H[end], D2, 1, d, d)
        reverse && reversempo!(H)
        return H
    end
end
hbathchain(N::Int, d::Int, e::Int, t::Int; tree=false) = hbathchain(N, d, (fill(e, N), fill(t, N-1), nothing); tree=tree)

function methylbluempo2(e1, e2, δ, N1, N2, N3, d1, d2, d3, S1a1, S2a1, S1a2, S2a2, cparS1S2)
    u = unitmat(3)

    c1 = only(S1a1[3])
    c2 = only(S2a2[3])
    c3 = only(cparS1S2[3])

    s2 = unitcol(1, 3)
    s1 = unitcol(2, 3)

    #Hs = e1*s1*s1' + e2*s2*s2' + δ*(s1*s2' + s2*s1')
    Hs = (e2-e1)*s2*s2' + δ*(s1*s2' + s2*s1') # e^(-is1*s1't)He^(is1*s1't)
    M = zeros(1,4,4,3,3,3)
    M[1,:,1,1,:,:] = up(Hs, c1*s1*s1', s2*s2', u)
    M[1,1,:,1,:,:] = up(Hs, c2*s2*s2', s1*s1', u)
    M[1,1,1,:,:,:] = up(Hs, c3*(s1*s2'+s2*s1'), u)

    H = TreeNetwork(Any[M])
    addtree!(H, 1, hbathchain(N1, d1, S1a1, S2a1; coupletox=true, tree=true))
    addtree!(H, 1, hbathchain(N2, d2, S2a2, S1a2; coupletox=true, tree=true))
    addtree!(H, 1, hbathchain(N3, d3, cparS1S2; coupletox=true, tree=true))
    return H
end

"""
    readchaincoeffs(fdir, params...)




"""
function readchaincoeffs(fdir, params...)
    n = length(params)
    dat = h5open(fdir, "r") do fid
        g = fid
        for i=1:n
            vars = keys(g)
            par = params[i]
            if typeof(par) <: Number
                ind = only(findall(x->x==par, parse.(Float64, vars)))
                g = g[vars[ind]]
            elseif typeof(par) <: String
                g = g[par]
            else
                throw(ArgumentError("parameter in position $i not valid"))
            end
        end
        return [read(g["e"]), read(g["t"]), read(g["c"])]
    end
    return dat
end
