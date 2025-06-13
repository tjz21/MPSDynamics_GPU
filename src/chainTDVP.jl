using cuTENSOR
import TensorOperations: CUDAAllocator, cuTENSORBackend
const GPU_ALLOCATOR = TensorOperations.CUDAAllocator{CUDA.default_memory, CUDA.default_memory, CUDA.default_memory}()

function initenvs(A::AbstractVector, M::AbstractVector, F::Nothing)
    N = length(A)
    F = Vector{Any}(undef, N+2)
    F[1] = fill!(similar(M[1], (1,1,1)), 1)
    F[N+2] = fill!(similar(M[1], (1,1,1)), 1)
    for k = N:-1:1
        F[k+1] = updaterightenv(A[k], M[k], F[k+2])
    end
    return F
end
initenvs(A::AbstractVector, M::AbstractVector) = initenvs(A, M, nothing)
initenvs(A::AbstractVector, M::AbstractVector, F::AbstractVector) = F

initrightenvs(A, M, F) = initenvs(A, M, F)
initrightenvs(A, M) = initenvs(A, M)

function initleftenvs(A::AbstractVector, M::AbstractVector, F::Nothing)
    N = length(A)
    F = Vector{Any}(undef, N+2)
    F[1] = fill!(similar(M[1], (1,1,1)), 1)
    F[N+2] = fill!(similar(M[1], (1,1,1)), 1)
    for k=1:N
        F[k+1] = updateleftenv(A[k], M[k], F[k])
    end
    return F
end
initleftenvs(A::AbstractVector, M::AbstractVector) = initleftenvs(A, M, nothing)
initleftenvs(A::AbstractVector, M::AbstractVector, F::AbstractVector) = F


function initrightenvs_full(A::AbstractVector, M::AbstractVector; Dplusmax=nothing, SVD=false)
    N = length(A)
    F = Vector{Any}(undef, N-1)
    Afull = Vector{Any}(undef, N-1)
    FR = fill!(similar(M[1], (1,1,1)), 1)
    for k = N:-1:2
        aleft, aright, d = size(A[k])
        C, AR = LQ_full(A[k]; SVD=SVD)
        Dmax = (Dplusmax != nothing ? min(aright*d, aleft+Dplusmax) : aright*d)
        AR = AR[1:Dmax,:,:]
        Afull[k-1] = AR
        F[k-1] = updaterightenv(AR, M[k], FR)
        C = nothing
        FR = F[k-1][1:aleft,:,1:aleft]
    end
    return F, Afull
end
function initleftenvs_full(A::AbstractVector, M::AbstractVector)
    N = length(A)
    F = Vector{Any}(undef, N-1)
    Afull = Vector{Any}(undef, N-1)
    FL = fill!(similar(M[1], (1,1,1)), 1)
    for k = 1:N-1
        aleft, aright, d = size(A[k])
        AL, C = QR_full(A[k])
        Afull[k] = AL
        F[k] = updaterightenv(AL, M[k], FL)
        C = nothing
        FL = F[k][1:aright,:,1:aright]
    end
    return F, Afull
end

function tdvp1rightsweep!(dt, A::AbstractVector, Afull::AbstractVector, M::AbstractVector, FR::AbstractVector; verbose=false, SVD=false, kwargs...)
    N = length(A)
    FLs = Vector{Any}(undef, N-1)
    AC = A[1]
    FL = fill!(similar(M[1], (1,1,1)), 1)
    for k = 1:N-1
        Dnew = size(FR[k])[1]
        Dold = size(AC)[2]
        
        AC, info = evolveAC(dt, AC, M[k], FL, FR[k], verbose; kwargs...)
        verbose && println("Sweep L->R: AC site $k, energy $(info[1])")
        verbose && Dnew!=Dold && println("*BondDimension $k-$(k+1) changed from $Dold to $Dnew")
        verbose && Dnew==Dold && println("*BondDimension $k-$(k+1) constant at $Dnew")

        AL, C = QR(AC; SVD=SVD)
        A[k] = AL
        FLs[k] = updateleftenv(AL, M[k], FL)
        FL = FLs[k]
        
        C, info = evolveC(dt, C, FL, FR[k], verbose; kwargs...)
        verbose && println("Sweep L->R: C between site $k and $(k+1), energy $(info[1])")

        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR AC[:] := C[-1,1] * Afull[k][1:Dnew,:,:][1,-2,-3]
        C = nothing

    end
    FR = fill!(similar(M[1], (1,1,1)), 1)
    AC, info = evolveAC(dt, AC, M[N], FL, FR, verbose; kwargs...)
    verbose && println("Sweep L->R: AC site $N, energy $(info[1])")
    A[N] = AC
    return A, FLs
end
function tdvp1leftsweep!(dt, A::AbstractVector, M::AbstractVector, FL::AbstractVector, Dlim::Int; SVD=false, verbose=false, Dplusmax=nothing, kwargs...)
    N = length(A)
    FRs = Vector{Any}(undef, N-1)
    Afull = Vector{Any}(undef, N-1)
    AC = A[N]
    FR = fill!(similar(M[1], (1,1,1)), 1)
    for k=N:-1:2
        Dnew = size(FL[k-1])[1]
        Dold, Dr, d = size(AC)
        Dmax = min(Dlim, (Dplusmax != nothing ? min(Dr*d, Dold+Dplusmax) : Dr*d))

        AC, info = evolveAC(dt, AC, M[k], FL[k-1], FR, verbose; kwargs...)
        verbose && println("Sweep R->L: AC site $k, energy $(info[1])")

        C, AR= LQ_full(AC; SVD=SVD)
        A[k] = AR[1:Dnew,:,:]
        
        FRs[k-1] = updaterightenv(AR[1:Dmax,:,:], M[k], FR) 
        FR = FRs[k-1][1:Dnew,:,1:Dnew]
        
        C, info = evolveC(dt, C, FL[k-1], FR, verbose; kwargs...)
        verbose && println("Sweep R->L: C between site $k and $(k-1), energy $(info[1])")
        
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR AC[:] := C[2,-2] * A[k-1][-1,2,-3]
        C = nothing

        Afull[k-1] = AR
    end
    FL = fill!(similar(M[1], (1,1,1)), 1)
    AC, info = evolveAC(dt, AC, M[1], FL, FR, verbose; kwargs...)
    verbose && println("Sweep R->L: AC site 1, energy $(info[1])")
    A[1] = AC
    return A, Afull, FRs
end

function updaterightbonds(FRs, A, M, th, Dlim, Dplusmax=nothing, SVD=false)
    N = length(A)
    olddims = bonddimsmps(A)
    ACs = Vector{Any}(undef, N)
    PAs = Vector{Any}(undef, N)
    PCs = Vector{Any}(undef, N-1)
    LA = fill!(similar(M[1], (1,1,1)), 1)
    AC = A[1]
    ACs[1] = AC
    # calculate PAs and PCs
    for k=1:N-1
        Dl = olddims[k]
        Dr = olddims[k+1]
        d = size(A[k])[3]
        Dmax = min((Dplusmax != nothing ? min(Dl*d, Dr+Dplusmax) : Dl*d), Dlim)
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR L[a,s,b,c] := LA[a,b',c'] * M[k][b',b,s,s'] * A[k][c',c,s']
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR PAs[k][a,s,a'] := L[a,s,b',c'] * FRs[k][:,:,1:Dr][a',b',c']
        AL, C = QR_full(AC; SVD=SVD)
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR AC[a,b,s] := C[a,a'] * A[k+1][1:Dr,:,:][a',b,s]
        C = nothing
        ACs[k+1] = AC
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR LA[a,b,c] := L[1:Dl,:,:,:][a',s,b,c] * AL[:,1:Dmax,:][a',a,s]
        L = nothing
        AL = nothing
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR PCs[k][a,a'] := LA[a,b',c'] * FRs[k][:,:,1:Dr][a',b',c']
    end
    Dl = olddims[N]
    @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR L[a,s,b,c] := LA[a,b',c'] * M[N][b',b,s,s'] * A[N][c',c,s']
    FR = fill!(similar(M[1], (1,1,1)), 1)
    @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR PAs[N][a,s,a'] := L[a,s,b',c'] * FR[a',b',c']

    # update bond-dimensions
    newdims, effect, acc = updatedims(A, PAs, PCs, th, Dlim)
    # construct FRs for next TDVP sweep
    for k=1:N-1
        Dnew = newdims[k+1]
        FRs[k] = FRs[k][1:Dnew,:,1:Dnew]
    end
    return FRs, ACs, effect, acc, Dims(newdims)
end

function updateleftbonds(FLs, A, M, th, Dlim)
    N = length(A)
    olddims = bonddimsmps(A)
    ACs = Vector{Any}(undef, N)
    PAs = Vector{Any}(undef, N)
    PCs = Vector{Any}(undef, N-1)
    RA = fill!(similar(M[1], (1,1,1)), 1)
    AC = A[N]
    ACs[N] = AC
    # calculate PAs and PCs
    for k=N:-1:2
        Dl = olddims[k]
        Dr = olddims[k+1]
        Dl == size(A[k])[1] || throw(ErrorException("Dl mismatch"))
        Dr == size(A[k])[2] || throw(ErrorException("Dr mismatch"))
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR R[a,s,b,c] := RA[a,b',c'] * M[k][b,b',s,s'] * A[k][c,c',s']
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR PAs[k][a',s,a] := R[a,s,b',c'] * FLs[k-1][:,:,1:Dl][a',b',c']
        C, AR = LQ_full(AC)
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR AC[a,b,s] := C[b',b] * A[k-1][:,1:Dl,:][a,b',s]
        C = nothing
        ACs[k-1] = AC
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR RA[a,b,c] := R[1:Dr,:,:,:][a',s,b,c] * AR[a,a',s]
        R = nothing
        AR = nothing
        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR PCs[k-1][a,a'] := RA[a,b',c'] * FLs[k-1][:,:,1:Dl][a',b',c']
    end
    Dr = olddims[2]
    @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR R[a,s,b,c] := RA[a,b',c'] * M[1][b,b',s,s'] * A[1][c,c',s']
    FL = fill!(similar(M[1], (1,1,1)), 1)
    @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR PAs[1][a',s,a] := R[a,s,b',c'] * FL[a',b',c']
    
    # update bond-dimensions
    newdims, effect, acc = updatedims(A, PAs, PCs, th, Dlim)
    # construct FLs for next TDVP sweep
    for k=1:N-1
        Dnew = newdims[k+1]
        FLs[k] = FLs[k][1:Dnew,:,1:Dnew]
    end
    return FLs, ACs, effect, acc, Dims(newdims)
end

function updatedims(A::AbstractVector, PAs::AbstractVector, PCs::AbstractVector, th, Dlim)
    N = length(PAs)
    olddims = bonddimsmps(A)
    newdims = Vector{Int}(undef, N+1)
    newdims[1] = newdims[N+1] = 1
    effect = Vector{Any}(undef, N-1)
    acc=0
    for k=1:N-1
        Dmax = min(size(PAs[k])[3], size(PAs[k+1])[1], newdims[k]*size(PAs[k])[2])

        effect[k] = Float64[norm(PAs[k][:,:,1:i])^2 - norm(PCs[k][1:i,1:i])^2 + norm(PAs[k+1][1:i,:,:])^2 for i=1:Dmax]
        
        Dnew=1
        while Dnew < Dmax && Dnew < Dlim
            x = (effect[k][Dnew+1]/effect[k][Dnew]) - 1
            if x > th
                Dnew+=1
            else
                break
            end
        end

        newdims[k+1] = max(olddims[k+1], Dnew)

        acc += norm(PAs[k][1:newdims[k],:,1:newdims[k+1]])^2
        acc -= norm(PCs[k][1:newdims[k+1],1:newdims[k+1]])^2
    end
    acc += norm(PAs[N][1:newdims[N],:,1:newdims[N+1]])^2
    return Dims(newdims), effect, acc
end

"""
    tdvp1sweep!(dt2, A::AbstractVector, M::AbstractVector, F=nothing; verbose=false, kwargs...)

Propagates the MPS A with the MPO M following the 1-site TDVP method. The sweep is done back and forth with a time step dt2/2. F represents the merged left and right parts of the site being propagated.  
"""

function tdvp1sweep!(dt2, A::AbstractVector, M::AbstractVector, F=nothing; verbose=false, kwargs...)
    N = length(A)
    dt = dt2/2
    F = initenvs(A, M, F)
    AC = A[1]
    for k = 1:N-1
        AC, info = evolveAC(dt, AC, M[k], F[k], F[k+2], verbose; kwargs...)
        verbose && println("Sweep L->R: AC site $k, energy $(info[1])")

        AL, C = QR(AC, 2)
        A[k] = AL
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        C, info = evolveC(dt, C, F[k+1], F[k+2], verbose; kwargs...)
        verbose && println("Sweep L->R: C between site $k and $(k+1), energy $(info[1])")

        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR AC[:] := C[-1,1] * A[k+1][1,-2,-3]
        C = nothing
    end
    k = N
    AC, info = evolveAC(dt2, AC, M[k], F[k], F[k+2], verbose; kwargs...)
    verbose && println("Sweep L->R: AC site $k, energy $(info[1])")
    for k = N-1:-1:1
        AR, C = QR(AC, 1)
        A[k+1] = AR
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        C, info = evolveC(dt, C, F[k+1], F[k+2], verbose; kwargs...)
        verbose && println("Sweep R->L: C between site $k and $(k+1), energy $(info[1])")

        @tensor backend=cuTENSORBackend() allocator=GPU_ALLOCATOR AC[:] := A[k][-1,1,-3] * C[1,-2]
        C = nothing

        AC, info = evolveAC(dt, AC, M[k], F[k], F[k+2], verbose; kwargs...)
        verbose && println("Sweep R->L: AC site $k, energy $(info[1])")
    end
    A[1] = AC
    return A, F
end
