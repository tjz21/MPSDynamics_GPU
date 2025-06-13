using CUDA, cuTENSOR, TensorOperations

import KrylovKit: exponentiate

"""
    exponentiate!(A_op, t::Number, x_target::CuArray; tol = 1e-12, maxiter = 250, kwargs...)

Evaluate e ** (t * A_op) * x_target with a power series.

`A_op` can be an arbitrary linear map, it should impliment a __call__ method.
The key to making this efficient is not recalculating the powers of `A_op` and
the factorial at every iteration. `t` is separate from `A_op` so that `A_op` can
remain real if `t` is complex.
The power series has the form

    x_target + t * A_op(x_target) + t ** 2 * A_op ** 2(x_target) / 2 + t ** 3 * A_op ** 3(x_target) / 6 + ...
"""
function exponentiate!(A_op, t::Number, x_target::CuArray; tol = 1e-12, maxiter = 250, kwargs...)
    iter_count = 0 
    converged = false

    current_term_generator = deepcopy(x_target) # stores (A^k x_0)/k! part, starts with (A^0 x_0)/0!

    term_to_add_to_sum = similar(x_target)

    current_t_power_k = t

    last_added_term_norm = 0.0

    while iter_count < maxiter
        A_op_output = A_op(current_term_generator)

        A_op_output ./= (iter_count + 1)
        
        current_term_generator = A_op_output # Update to the new (A_op^(iter_count+1) x_0) / (iter_count+1)!

        copyto!(term_to_add_to_sum, current_term_generator)
        rmul!(term_to_add_to_sum, current_t_power_k)

        x_target .+= term_to_add_to_sum

        # Convergence Check
        last_added_term_norm = norm(term_to_add_to_sum)
        x_target_norm = norm(x_target)
            
        if x_target_norm > tol 
            if last_added_term_norm < tol * x_target_norm
                converged = true
                break
            end
        elseif last_added_term_norm < tol 
             converged = true
             break
        end
        
        iter_count += 1
        current_t_power_k *= t 
    end

    return (converged = converged, iterations = iter_count, residual = last_added_term_norm)
end

function ACOAC(AC::AbstractArray{T1, 3}, O::AbstractArray{T2, 2}) where {T1,T2}
    @tensor v = tensorscalar(conj(AC[a,b,s']) * O[s',s] * AC[a,b,s])
end

function contractC!(A::CuArray{T1,2}, C::CuArray{T2,2}, dir::Int) where {T1,T2}
    A_new = @tensor A[a0,s] := A[a0',s] * C[a0,a0']
    return A_new
end
function contractC!(A::CuArray{T1,3}, C::CuArray{T2,2}, dir::Int) where {T1,T2}
    local A_new
    if dir==1
        A_new = @tensor A[a0,a1,s] := A[a0',a1,s] * C[a0,a0']
    elseif dir==2
        A_new = @tensor A[a0,a1,s] := A[a0,a1',s] * C[a1,a1']
    else
        throw("invalid argument dir=$dir")
    end
    return A_new
end
function contractC!(A::CuArray{T1,4}, C::CuArray{T2,2}, dir::Int) where {T1,T2}
    local A_new
    if dir==1
        A_new = @tensor A[a0,a1,a2,s] := A[a0',a1,a2,s] * C[a0,a0']
    elseif dir==2
        A_new = @tensor A[a0,a1,a2,s] := A[a0,a1',a2,s] * C[a1,a1']
    elseif dir==3
        A_new = @tensor A[a0,a1,a2,s] := A[a0,a1,a2',s] * C[a2,a2']
    else
        throw("invalid argument dir=$dir")
    end
    return A_new
end
function contractC!(A::CuArray{T1,5}, C::CuArray{T2,2}, dir::Int) where {T1,T2}
    local A_new
    if dir==1
        A_new = @tensor A[a0,a1,a2,a3,s] := A[a0',a1,a2,a3,s] * C[a0,a0']
    elseif dir==2
        A_new = @tensor A[a0,a1,a2,a3,s] := A[a0,a1',a2,a3,s] * C[a1,a1']
    elseif dir==3
        A_new = @tensor A[a0,a1,a2,a3,s] := A[a0,a1,a2',a3,s] * C[a2,a2']
    elseif dir==4
        A_new = @tensor A[a0,a1,a2,a3,s] := A[a0,a1,a2,a3',s] * C[a3,a3']
    else
        throw("invalid argument dir=$dir")
    end
    return A_new
end
function contractC!(A::CuArray{T1,6}, C::CuArray{T2,2}, dir::Int) where {T1,T2}
    local A_new
    if dir==1
        A_new = @tensor A[a0,a1,a2,a3,a4,s] := A[a0',a1,a2,a3,a4,s] * C[a0,a0']
    elseif dir==2
        A_new = @tensor A[a0,a1,a2,a3,a4,s] := A[a0,a1',a2,a3,a4,s] * C[a1,a1']
    elseif dir==3
        A_new = @tensor A[a0,a1,a2,a3,a4,s] := A[a0,a1,a2',a3,a4,s] * C[a2,a2']
    elseif dir==4
        A_new = @tensor A[a0,a1,a2,a3,a4,s] := A[a0,a1,a2,a3',a4,s] * C[a3,a3']
    elseif dir==5
        A_new = @tensor A[a0,a1,a2,a3,a4,s] := A[a0,a1,a2,a3,a4',s] * C[a4,a4']
    else
        throw("invalid argument dir=$dir")
    end
    return A_new
end

function rhoAAstar(ρ::CuArray{T1,2}, A::CuArray{T2,2}, indir::Int) where {T1,T2}
    @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,s]) * A[b0,s])
end
function rhoAAstar(ρ::CuArray{T1,2}, A::CuArray{T2,2}) where {T1,T2}
    @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,s]) * A[b0,s])
end
function rhoAAstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}, indir::Int, outdir::Int) where {T1,T2}
    indir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,s]) * A[b0,b,s]
    indir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,a0,s]) * A[b,b0,s]
    throw("invalid arguments indir=$indir, outdir=$outdir")
end
function rhoAAstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}) where {T1,T2}
    @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,s]) * A[b0,b,s]
end
function rhoABstar(ρ::CuArray{T1,2}, A::CuArray{T2,2}, B::CuArray{T3,2}) where {T1,T2,T3}
    @tensor ρO = tensorscalar(ρ[a0,b0] * conj(B[a0,s]) * A[b0,s])
end
function rhoABstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}, B::CuArray{T3,3}) where {T1,T2,T3}
    @tensor ρO[a,b] := ρ[a0,b0] * conj(B[a0,a,s]) * A[b0,b,s]
end

function rhoAAstar(ρ::CuArray{T1,2}, A::CuArray{T2,4}, indir::Int, outdir::Int) where {T1,T2}
    indir==1 && outdir==2 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,c0,s]) * A[b0,b,c0,s]
    indir==1 && outdir==3 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a0,c0,a,s]) * A[b0,c0,b,s]

    indir==2 && outdir==1 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a,a0,c0,s]) * A[b,b0,c0,s]
    indir==2 && outdir==3 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,a0,a,s]) * A[c0,b0,b,s]

    indir==3 && outdir==1 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a,c0,a0,s]) * A[b,c0,b0,s]
    indir==3 && outdir==2 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,a,a0,s]) * A[c0,b,b0,s]
    throw("invalid arguments indir=$indir, outdir=$outdir")
end
function rhoAAstar(ρ::CuArray{T1,2}, A::CuArray{T2,5}, indir::Int, outdir::Int) where {T1,T2}
    indir==1 && outdir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,c0,c1,s]) * A[b0,b,c0,c1,s]
    indir==1 && outdir==3 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a0,c0,a,c1,s]) * A[b0,c0,b,c1,s]
    indir==1 && outdir==4 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a0,c0,c1,a,s]) * A[b0,c0,c1,b,s]

    indir==2 && outdir==1 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a,a0,c0,c1,s]) * A[b,b0,c0,c1,s]
    indir==2 && outdir==3 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,a0,a,c1,s]) * A[c0,b0,b,c1,s]
    indir==2 && outdir==4 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,a0,c1,a,s]) * A[c0,b0,c1,b,s]

    indir==3 && outdir==1 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a,c0,a0,c1,s]) * A[b,c0,b0,c1,s]
    indir==3 && outdir==2 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,a,a0,c1,s]) * A[c0,b,b0,c1,s]
    indir==3 && outdir==4 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,c1,a0,a,s]) * A[c0,c1,b0,b,s]

    indir==4 && outdir==1 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[a,c0,c1,a0,s]) * A[b,c0,c1,b0,s]
    indir==4 && outdir==2 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,a,c1,a0,s]) * A[c0,b,c1,b0,s]
    indir==4 && outdir==3 && return @tensor opt=true ρO[a,b] := ρ[a0,b0] * conj(A[c0,c1,a,a0,s]) * A[c0,c1,b,b0,s]
    throw("invalid arguments indir=$indir, outdir=$outdir")
end

function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,2}, O::AbstractArray{T3,2}, indir::Int, ::Nothing) where {T1,T2,T3}
    @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,s']) * O[s',s] * A[b0,s])
end
function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,2}, O::AbstractArray{T3,2}, ::Nothing) where {T1,T2,T3}
    @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,s']) * O[s',s] * A[b0,s])
end
function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}, O::AbstractArray{T3,2}, indir::Int, ::Nothing) where {T1,T2,T3}
    indir==1 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,c0,s']) * O[s',s] * A[b0,c0,s])
    indir==2 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[c0,a0,s']) * O[s',s] * A[c0,b0,s])
end
function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}, O::AbstractArray{T3,2}, ::Nothing) where {T1,T2,T3}
    @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,c0,s']) * O[s',s] * A[b0,c0,s])
end

function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,4}, O::AbstractArray{T3,2}, indir::Int, ::Nothing) where {T1,T2,T3}
    indir==1 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,c0,c1,s']) * O[s',s] * A[b0,c0,c1,s])
    indir==2 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[c0,a0,c1,s']) * O[s',s] * A[c0,b0,c1,s])
    indir==3 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[c0,c1,a0,s']) * O[s',s] * A[c0,c1,b0,s])
end
function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,5}, O::AbstractArray{T3,2}, indir::Int, ::Nothing) where {T1,T2,T3}
    indir==1 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[a0,c0,c1,c2,s']) * O[s',s] * A[b0,c0,c1,c2,s])
    indir==2 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[c0,a0,c1,c2,s']) * O[s',s] * A[c0,b0,c1,c2,s])
    indir==3 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[c0,c1,a0,c2,s']) * O[s',s] * A[c0,c1,b0,c2,s])
    indir==4 && return @tensor ρO = tensorscalar(ρ[a0,b0] * conj(A[c0,c1,c2,a0,s']) * O[s',s] * A[c0,c1,c2,b0,s])
end

function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}, O::AbstractArray{T3,2}, indir::Int, outdir::Int) where {T1,T2,T3}
    indir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,s']) * O[s',s] * A[b0,b,s]
    indir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,a0,s']) * O[s',s] * A[b,b0,s]
    throw("invalid arguments indir=$indir, outdir=$outdir")
end
function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,3}, O::AbstractArray{T3,2}) where {T1,T2,T3}
    @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,s']) * O[s',s] * A[b0,b,s]
end

function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,4}, O::AbstractArray{T3,2}, indir::Int, outdir::Int) where {T1,T2,T3}
    indir==1 && outdir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,c0,s']) * O[s',s] * A[b0,b,c0,s]
    indir==1 && outdir==3 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,c0,a,s']) * O[s',s] * A[b0,c0,b,s]

    indir==2 && outdir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,a0,c0,s']) * O[s',s] * A[b,b0,c0,s]
    indir==2 && outdir==3 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,a0,a,s']) * O[s',s] * A[c0,b0,b,s]

    indir==3 && outdir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,c0,a0,s']) * O[s',s] * A[b,c0,b0,s]
    indir==3 && outdir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,a,a0,s']) * O[s',s] * A[c0,b,b0,s]
    throw("invalid arguments indir=$indir, outdir=$outdir")
end
function rhoAOAstar(ρ::CuArray{T1,2}, A::CuArray{T2,5}, O::AbstractArray{T3,2}, indir::Int, outdir::Int) where {T1,T2,T3}
    indir==1 && outdir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,a,c0,c1,s']) * O[s',s] * A[b0,b,c0,c1,s]
    indir==1 && outdir==3 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,c0,a,c1,s']) * O[s',s] * A[b0,c0,b,c1,s]
    indir==1 && outdir==4 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a0,c0,c1,a,s']) * O[s',s] * A[b0,c0,c1,b,s]

    indir==2 && outdir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,a0,c0,c1,s']) * O[s',s] * A[b,b0,c0,c1,s]
    indir==2 && outdir==3 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,a0,a,c1,s']) * O[s',s] * A[c0,b0,b,c1,s]
    indir==2 && outdir==4 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,a0,c1,a,s']) * O[s',s] * A[c0,b0,c1,b,s]

    indir==3 && outdir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,c0,a0,c1,s']) * O[s',s] * A[b,c0,b0,c1,s]
    indir==3 && outdir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,a,a0,c1,s']) * O[s',s] * A[c0,b,b0,c1,s]
    indir==3 && outdir==4 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,c1,a0,a,s']) * O[s',s] * A[c0,c1,b0,b,s]

    indir==4 && outdir==1 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[a,c0,c1,a0,s']) * O[s',s] * A[b,c0,c1,b0,s]
    indir==4 && outdir==2 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,a,c1,a0,s']) * O[s',s] * A[c0,b,c1,b0,s]
    indir==4 && outdir==3 && return @tensor ρO[a,b] := ρ[a0,b0] * conj(A[c0,c1,a,a0,s']) * O[s',s] * A[c0,c1,b,b0,s]
    throw("invalid arguments indir=$indir, outdir=$outdir")
end

#dir gives the open index, where dir=1 is the first child
#Fs must be given in the correct order
#left = direction of parent
#right = direction of children
function updateleftenv(A::CuArray{T1,2}, M::CuArray{T2,3}, dir::Int) where {T1,T2}
    @tensor F[a,b,c] := conj(A[a,s'])*M[b,s',s]*A[c,s]
    return F
end
function updaterightenv(A::CuArray{T1,2}, M::CuArray{T2,3}) where {T1,T2}
    @tensor F[a,b,c] := conj(A[a,s'])*M[b,s',s]*A[c,s]
    return F
end
function updateleftenv(A::CuArray{T1,3}, M::CuArray{T2,4}, FL) where {T1,T2}
    @tensor F[a,b,c] := FL[a0,b0,c0]*conj(A[a0,a,s'])*M[b0,b,s',s]*A[c0,c,s]
    return F
end
function updateleftenv(A::CuArray{T1,3}, M::CuArray{T2,4}, dir::Int, F0) where {T1,T2}
    @tensor F[a,b,c] := F0[a0,b0,c0]*conj(A[a0,a,s'])*M[b0,b,s',s]*A[c0,c,s]
    return F
end
function updaterightenv(A::CuArray{T1,3}, M::CuArray{T2,4}, FR) where {T1,T2}
    @tensor F[a,b,c] := FR[a0,b0,c0]*conj(A[a,a0,s'])*M[b,b0,s',s]*A[c,c0,s]
    return F
end

function updateleftenv(A::CuArray{T1,4}, M::CuArray{T2,5}, dir::Int, F0, F1) where {T1,T2}
    dir==1 && (Aperm = [1,3,2,4]; Mperm = [1,3,2,4,5])
    dir==2 && (Aperm = [1,2,3,4]; Mperm = [1,2,3,4,5])
    At = permutedims(A, Aperm)
    Mt = permutedims(M, Mperm)
    @tensor opt = true F[a,b,c] := F0[a0,b0,c0]*F1[a1,b1,c1]*conj(At[a0,a1,a,s'])*Mt[b0,b1,b,s',s]*At[c0,c1,c,s]
    return F
end
function updaterightenv(A::CuArray{T1,4}, M::CuArray{T2,5}, F1, F2) where {T1,T2}
    @tensor opt=true F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*conj(A[a,a1,a2,s'])*M[b,b1,b2,s',s]*A[c,c1,c2,s]
    return F
end
function updateleftenv(A::CuArray{T1,5}, M::CuArray{T2,6}, dir::Int, F0, F1, F2) where {T1,T2}
    dir==1 && (Aperm = [1,3,4,2,5]; Mperm = [1,3,4,2,5,6])
    dir==2 && (Aperm = [1,2,4,3,5]; Mperm = [1,2,4,3,5,6])
    dir==3 && (Aperm = [1,2,3,4,5]; Mperm = [1,2,3,4,5,6])
    At = permutedims(A, Aperm)
    Mt = permutedims(M, Mperm)
    @tensor opt=true F[a,b,c] := F0[a0,b0,c0]*F1[a1,b1,c1]*F2[a2,b2,c2]*conj(At[a0,a1,a2,a,s'])*Mt[b0,b1,b2,b,s',s]*At[c0,c1,c2,c,s]
    return F
end
function updaterightenv(A::CuArray{T1,5}, M::CuArray{T2,6}, F1, F2, F3) where {T1,T2}
    @tensor opt=true F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*conj(A[a,a1,a2,a3,s'])*M[b,b1,b2,b3,s',s]*A[c,c1,c2,c3,s]
    return F
end

#dir gives the open index, where dir=1 is the parent
function updateenv(A, M, dir, F1)
    @tensor F[a,b,c] := conj(A[a,s'])*M[b,s',s]*A[c,s]
end
function updateenv(A, M, dir, F1, F2)
    if dir==1
        updateenv1(A, M, F2)
    else
        updateenv2(A, M, F1)
    end
end
function updateenv(A, M, dir, F1, F2, F3)
    if dir==1
        updateenv1(A, M, F2, F3)
    elseif dir==2
        updateenv2(A, M, F1, F3)
    else
        updateenv3(A, M, F1, F2)
    end
end
function updateenv(A, M, dir, F1, F2, F3, F4)
    if dir==1
        updateenv1(A, M, F2, F3, F4)
    elseif dir==2
        updateenv2(A, M, F1, F3, F4)
    elseif dir==3
        updateenv3(A, M, F1, F2, F4)
    else
        updateenv4(A, M, F1, F2, F3)
    end
end
function updateenv1(A, M, F1)
    @tensor F[a,b,c] := F1[a1,b1,c1]*conj(A[a,a1,s'])*M[b,b1,s',s]*A[c,c1,s]
    return F
end
function updateenv2(A, M, F1)
    @tensor F[a,b,c] := F1[a1,b1,c1]*conj(A[a1,a,s'])*M[b1,b,s',s]*A[c1,c,s]
    return F
end
function updateenv1(A, M, F1, F2)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*conj(A[a,a1,a2,s'])*M[b,b1,b2,s',s]*A[c,c1,c2,s]
    return F
end
function updateenv2(A, M, F1, F2)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*conj(A[a1,a,a2,s'])*M[b1,b,b2,s',s]*A[c1,c,c2,s]
    return F
end
function updateenv3(A, M, F1, F2)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*conj(A[a1,a2,a,s'])*M[b1,b2,b,s',s]*A[c1,c2,c,s]
    return F
end
function updateenv1(A, M, F1, F2, F3)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*conj(A[a,a1,a2,a3,s'])*M[b,b1,b2,b3,s',s]*A[c,c1,c2,c3,s]
    return F
end
function updateenv2(A, M, F1, F2, F3)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*conj(A[a1,a,a2,a3,s'])*M[b1,b,b2,b3,s',s]*A[c1,c,c2,c3,s]
    return F
end
function updateenv3(A, M, F1, F2, F3)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*conj(A[a1,a2,a,a3,s'])*M[b1,b2,b,b3,s',s]*A[c1,c2,c,c3,s]
    return F
end
function updateenv4(A, M, F1, F2, F3)
    @tensor F[a,b,c] := F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*conj(A[a1,a2,a3,a,s'])*M[b1,b2,b3,b,s',s]*A[c1,c2,c3,c,s]
    return F
end

function applyH2(AA, H1, H2, F1, F2)
    return @tensor opt=true HAA[a1,s1,a2,s2] := F1[a1,b1,c1]*AA[c1,s1',c2,s2']*H1[b1,b,s1,s1']*H2[b,b2,s2,s2']*F2[a2,b2,c2]
end
function applyH1(AC, M, F)
    return @tensor opt=true HAC[a,s'] := F[a,b,c]*AC[c,s]*M[b,s',s]
end
function applyH1(AC, M, F0, F1)
    return @tensor opt=true HAC[a0,a1,s'] := F0[a0,b0,c0]*F1[a1,b1,c1]*AC[c0,c1,s]*M[b0,b1,s',s]
end
function applyH1(AC, M, F0, F1, F2)
    return @tensor opt=true HAC[a0,a1,a2,s'] := F0[a0,b0,c0]*F1[a1,b1,c1]*F2[a2,b2,c2]*AC[c0,c1,c2,s]*M[b0,b1,b2,s',s]
end
function applyH1(AC, M, F0, F1, F2, F3)
    return @tensor opt=true HAC[a0,a1,a2,a3,s'] := F0[a0,b0,c0]*F1[a1,b1,c1]*F2[a2,b2,c2]*F3[a3,b3,c3]*AC[c0,c1,c2,c3,s]*M[b0,b1,b2,b3,s',s]
end
function applyH0(C, FL, FR)
    return @tensor HC[α,β] := FL[α,a,α']*C[α',β']*FR[β,a,β']
end

#sets the right/left bond dimension of A, ie will truncate if Dnew is smaller than the current bond dimension and zero pad if it's larger
function setrightbond(A, Dnew::Int)
    Dl, Dr, d  = size(A)
    a = fill!(similar(A, Dl, Dnew, d), 0)
    if Dr > Dnew
        a = A[:,1:Dnew,:]
    else
        a[:,1:Dr,:] = A
    end
    return a
end
function setleftbond(A, Dnew::Int)
    Dl, Dr, d  = size(A)
    a = fill!(similar(A, Dnew, Dr, d), 0)
    if Dl > Dnew
        a = A[1:Dnew,:,:]
    else
        a[1:Dl,:,:] = A
    end
    return a
end

function setbond(A::AbstractArray{T, 2}, Dnew::Int) where T
    Dold, d = size(A)
    a = fill!(similar(A, Dnew, d), 0)
    Dold > Dnew ? (D = Dnew) : (D = Dold)
    a[1:D,:] = A[1:D,:]
    return a
end
function setbond(A::AbstractArray{T, 2}) where T
    return A
end
function setbond(A::AbstractArray{T, 3}, D1new::Int, D2new::Int) where T
    D1old, D2old, d = size(A)
    a = fill!(similar(A, D1new, D2new, d), 0)
    D1old > D1new ? (D1 = D1new) : (D1 = D1old)
    D2old > D2new ? (D2 = D2new) : (D2 = D2old)
    a[1:D1,1:D2,:] = A[1:D1,1:D2,:]
    return a
end
function setbond(A::AbstractArray{T, 3}, D2new::Int) where T
    D1old, D2old, d = size(A)
    a = fill!(similar(A, D1old, D2new, d), 0)
    D2old > D2new ? (D2 = D2new) : (D2 = D2old)
    a[:,1:D2,:] = A[:,1:D2,:]
    return a
end
function setbond(A::AbstractArray{T, 4}, D1new::Int, D2new::Int, D3new::Int) where T
    D1old, D2old, D3old, d = size(A)
    a = fill!(similar(A, D1new, D2new, D3new, d), 0)
    D1old > D1new ? (D1 = D1new) : (D1 = D1old)
    D2old > D2new ? (D2 = D2new) : (D2 = D2old)
    D3old > D3new ? (D3 = D3new) : (D3 = D3old)
    a[1:D1,1:D2,1:D3,:] = A[1:D1,1:D2,1:D3,:]
    return a
end
function setbond(A::AbstractArray{T, 4}, D2new::Int, D3new::Int) where T
    D1old, D2old, D3old, d = size(A)
    a = fill!(similar(A, D1old, D2new, D3new, d), 0)
    D2old > D2new ? (D2 = D2new) : (D2 = D2old)
    D3old > D3new ? (D3 = D3new) : (D3 = D3old)
    a[:,1:D2,1:D3,:] = A[:,1:D2,1:D3,:]
    return a
end
function setbond(A::AbstractArray{T, 5}, D1new::Int, D2new::Int, D3new::Int, D4new::Int) where T
    D1old, D2old, D3old, D4old, d = size(A)
    a = fill!(similar(A, D1new, D2new, D3new, D4new, d), 0)
    D1old > D1new ? (D1 = D1new) : (D1 = D1old)
    D2old > D2new ? (D2 = D2new) : (D2 = D2old)
    D3old > D3new ? (D3 = D3new) : (D3 = D3old)
    D4old > D4new ? (D4 = D4new) : (D4 = D4old)
    a[1:D1,1:D2,1:D3,1:D4,:] = A[1:D1,1:D2,1:D3,1:D4,:]
    return a
end
function setbond(A::AbstractArray{T, 5}, D2new::Int, D3new::Int, D4new::Int) where T
    D1old, D2old, D3old, D4old, d = size(A)
    a = fill!(similar(A, D1old, D2new, D3new, D4new, d), 0)
    D2old > D2new ? (D2 = D2new) : (D2 = D2old)
    D3old > D3new ? (D3 = D3new) : (D3 = D3old)
    D4old > D4new ? (D4 = D4new) : (D4 = D4old)
    a[:,1:D2,1:D3,1:D4,:] = A[:,1:D2,1:D3,1:D4,:]
    return a
end

function evolveAC2(dt::Float64, A1, A2, M1, M2, FL, FR, energy=false; kwargs...)
    @tensor AA[a,sa,b,sb] := A1[a,c,sa] * A2[c,b,sb]

    tol = get(kwargs, :tol, 1e-12)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)

    AAnew, info = exponentiate!(x->applyH2(x, M1, M2, FL, FR), -im*dt, AA; tol=tol, krylovdim=krylovdim, maxiter=maxiter)

    if energy
        E = real(dot(AAnew, applyH2(AAnew, M1, M2, FL, FR)))
        return AAnew, (E, info)
    end
    return AAnew, info
end

function evolveAC(dt::Float64, AC, M, FL, FR, energy=false; projerr = false, kwargs...)

    Dlnew, w, Dl = size(FL)
    Drnew, w, Dr = size(FR)

    AC = setrightbond(AC, Drnew)
    AC = setleftbond(AC, Dlnew)

    tol = get(kwargs, :tol, 1e-12)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)

    if projerr
        pe = norm(applyH1(AC, M, FL, FR))
    else
        pe = nothing
    end
    
    AC, expinfo = exponentiate!(x->applyH1(x, M, FL, FR), -im*dt, AC; tol=tol, krylovdim=krylovdim, maxiter=maxiter)

    if energy
        E = real(dot(AC, applyH1(AC, M, FL, FR)))
    else
        E = nothing
    end
    
    return AC, (E, pe, expinfo)
end

function evolveC(dt::Float64, C, FL, FR, energy=false; projerr = false, kwargs...)

    tol = get(kwargs, :tol, 1e-12)
    krylovdim = get(kwargs, :krylovdim, 30)
    maxiter = get(kwargs, :maxiter, 100)

    if projerr
        pe = norm(applyH0(C, FL, FR))
    else
        pe = nothing
    end
        
    C, expinfo = exponentiate!(x->applyH0(x, FL, FR), im*dt, C; tol=tol, krylovdim=krylovdim, maxiter=maxiter)

    if energy
        E = real(dot(C, applyH0(C, FL, FR)))
    else
        E = nothing
    end
    
    return C, (E, pe, expinfo)
end

import LinearAlgebra: transpose
function transpose(A::AbstractArray, dim1::Int, dim2::Int)
    nd=ndims(A)
    perm=collect(1:nd)
    perm[dim1]=dim2
    perm[dim2]=dim1
    permutedims(A, perm)
end
function QR(A::AbstractArray, i::Int)
    dims = [size(A)...]
    nd = length(dims)
    ds = collect(1:nd)

    permuted_A = permutedims(A, circshift(ds, -i))
    reshaped_input_for_qr = reshape(permuted_A, :, dims[i])
    AL_factor, C = qr(reshaped_input_for_qr)

    local AL_to_return

    if A isa CuArray
        explicit_Q_on_gpu = CuMatrix(AL_factor)
        reshaped_Q_on_gpu = reshape(explicit_Q_on_gpu, circshift(dims, -i)...)
        AL_to_return = permutedims(reshaped_Q_on_gpu, circshift(ds, i))
    else
        temp_cpu_matrix_from_al_factor = Matrix(AL_factor)
        AL_to_return = permutedims(reshape(temp_cpu_matrix_from_al_factor, circshift(dims, -i)...), circshift(ds, i))
    end
    return AL_to_return, C
end

function QR(A::AbstractArray; SVD=false)
    Dl, Dr, d = size(A)
    if !SVD
        Q_factor, R_val = qr(reshape(permutedims(A, [1,3,2]), Dl*d, Dr))
        
        local AL_to_return
        if A isa CuArray
            explicit_Q_on_gpu = CuMatrix(Q_factor)
            reshaped_Q_on_gpu = reshape(explicit_Q_on_gpu, Dl, d, Dr)
            AL_to_return = permutedims(reshaped_Q_on_gpu, [1,3,2])
        else
            temp_cpu_matrix_from_q_factor = Matrix(Q_factor)
            AL_to_return = permutedims(reshape(temp_cpu_matrix_from_q_factor, Dl, d, Dr), [1,3,2])
        end
        return AL_to_return, R_val
    else
        
        reshaped_A = permutedims(A, [1,3,2]) 
        matrix_for_svd = reshape(reshaped_A, Dl*d, Dr)

        F = svd(matrix_for_svd)

        AL = permutedims(reshape(F.U, Dl, d, Dr), [1,3,2])
        R = Diagonal(F.S)*F.Vt

        return AL, R
    end    
end

function QR_full(A::AbstractArray; SVD=false)
   Dl, Dr, d = size(A)
   if !SVD
       Q_factor, R = qr(reshape(permutedims(A, [1,3,2]), Dl*d, Dr))
       local AL_to_return
       if A isa CuArray
           temp_cpu_matrix_from_q_factor_full = Matrix(Q_factor)
           AL_cpu_version = permutedims(reshape(temp_cpu_matrix_from_q_factor_full, Dl, d, Dl*d), [1,3,2])
           AL_to_return = CuArray(AL_cpu_version)
       else
           temp_cpu_matrix_from_q_factor_full = Matrix(Q_factor)
           AL_to_return = permutedims(reshape(temp_cpu_matrix_from_q_factor_full, Dl, d, Dl*d), [1,3,2])
       end
       return AL_to_return, R #(R = C)
   else
       reshaped_A_full = permutedims(A, [1,3,2])
       matrix_for_svd_full = reshape(reshaped_A_full, Dl*d, Dr)

       F = svd(matrix_for_svd_full, full=true)

       AL = permutedims(reshape(F.U, Dl, d, Dr), [1,3,2])
       R = Diagonal(F.S)*F.Vt

       return AL, R
   end
end

function LQ_full(A::AbstractArray; SVD=false)
   Dl, Dr, d = size(A)
   if !SVD
       L_factor_val, Q_factor = lq(reshape(A, Dl, Dr*d))
       local AR_to_return
       if A isa CuArray
           temp_cpu_matrix_from_q_factor_lq = Matrix(Q_factor)
           AR_cpu_version = reshape(temp_cpu_matrix_from_q_factor_lq, Dr*d, Dr, d)
           AR_to_return = CuArray(AR_cpu_version)
       else
            temp_cpu_matrix_from_q_factor_lq = Matrix(Q_factor)
            AR_to_return = reshape(temp_cpu_matrix_from_q_factor_lq, Dr*d, Dr, d)
       end
       return L_factor_val, AR_to_return #(L = c)
   else

       matrix_for_svd_lq = reshape(A, Dl, Dr*d)

       F = svd(matrix_for_svd_lq, full=true)
      
       AR = reshape(F.Vt, Dr*d, Dr, d)
       L = F.U*Diagonal(F.S)

       return L, AR
   end
end
