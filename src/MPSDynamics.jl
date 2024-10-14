module MPSDynamics

GPU = false
if length(ARGS) > 0 && ARGS[1] == "GPU"

  # A CUDA compatible macro to overwrite TensorOperations.@tensoropt
  macro tensoropt(expressions...)
    if length(expressions) == 1
      ex = expressions[1]
      optdict = TensorOperations.optdata(ex)
    elseif length(expressions) == 2
      optex = expressions[1]
      ex = expressions[2]
      optdict = TensorOperations.optdata(optex, ex)
    end

    cuwrapdict = Dict{Any,Any}()
    parser = TensorOperations.TensorParser()
    parser.contractiontreebuilder = network -> TensorOperations.optimaltree(
      network, optdict
    )[1]
    parser.preprocessors[end] = ex -> TensorOperations.extracttensorobjects(
      ex, cuwrapdict
    )
    push!(
      parser.postprocessors, 
      ex -> TensorOperations.addcutensorwraps(ex, cuwrapdict)
    )
    return esc(parser(ex))
  end

  using CUDA
  # We also overwrite @tensor; this does most of the GPU acceleration for us.
  import TensorOperations: @cutensor as @tensor

  using LinearAlgebra

  # Make LQPackedQ functional on GPUs (see 
  # https://github.com/JuliaGPU/CUDA.jl/issues/1893#issuecomment-1533783048)
  function Base.getindex(
    Q::LinearAlgebra.LQPackedQ{<:Any, <:CuArray}, ::Colon, j::Int
  )
    y = CuArray{eltype(Q)}(diagm(size(Q, 2), 1, (-j + 1) => [1]))
    return lmul!(Q, y)
  end  
  function Base.getindex(
    Q::LinearAlgebra.LQPackedQ{<:Any, <:CuArray}, i::Int, j::Int
  )
    x = CuArray{eltype(Q)}(diagm(size(Q, 2), 1, (-j + 1) => [1]))
    y = CuArray{eltype(Q)}(diagm(size(Q, 2), 1, (-i + 1) => [1]))
    Q = lmul!(Q, y)
    return sum(Q * x)
  end
  function CuMatrix{T}(Q::LinearAlgebra.LQPackedQ{S}) where {T,S}
    return CuMatrix{T}(lmul!(
      Q, 
      CuMatrix{S}(I, size(Q, 1), min(size(Q.factors)...))
    ))
  end

  # Define CUDA versions of LAPACK LQ functions in terms of QR functions
  function LinearAlgebra.LAPACK.orglq!(A::CuArray, tau::CuArray)
    return CuArray(transpose(CUDA.CUSOLVER.orgqr!(
      CuArray(transpose(A)), tau 
    ))) 
  end
  function LinearAlgebra.LAPACK.ormlq!(
    side::Char, trans::Char, A::CuArray, tau::CuArray, C::Union{Vector, CuArray}
  )
    return CuArray(transpose(CUDA.CUSOLVER.ormqr!(
      side, trans, CuArray(transpose(A)), tau, CuArray(C)
    ))) 
  end
  # This version of lq returns an LQPackedQ
  function LinearAlgebra.lq!(A::CuArray{T, 2}) where T
    Q, R = qr(CuArray(transpose(A)))
    if typeof(Q) <: LinearAlgebra.QRCompactWYQ
      Q_factors, Q_t = Q.factors, Q.T
    elseif typeof(Q) <: LinearAlgebra.QRPackedQ
      Q_factors, Q_t = Q.factors, Q.τ
    else
      println("don't know what type Q is")
    end
    Q = LinearAlgebra.LQPackedQ(
      CuMatrix(transpose(Q_factors)),
      CuMatrix(transpose(Q_t))[1:end]
    )
    L = CuArray(transpose(R))
    return L, Q
  end

  # A simpler way to do lq, but returning L and Q instead of the packed matrices
  #function LinearAlgebra.lq!(A::CuMatrix{T}) where T 
  #  Q, R = qr(CuArray(transpose(A::CuMatrix{T})))
  #  Q = lmul!(
  #    Q, 
  #    CuArray{eltype(Q)}(diagm(size(Q)..., [1 for _ in 1:min(size(Q)...)]))
  #  )
  #  L, Q = CuArray(transpose(R)), CuArray(transpose(Q))
  #  return L, Q
  #end

  GPU = true
  println("Attempting to run on GPUs")
  # make sure to load cuTensor
  using cuTENSOR
else
  # Define CuArray and CuMatrix as dummy variables to avoid errors
  CuArray = Array
  CuMatrix = Matrix
end

using JLD, HDF5, Random, Dates, Plots, Printf, Distributed, LinearAlgebra, DelimitedFiles, KrylovKit, TensorOperations, GraphRecipes, SpecialFunctions, ITensors

include("fundamentals.jl")
include("reshape.jl")
include("tensorOps.jl")
include("measure.jl")
include("observables.jl")
include("logiter.jl")
include("flattendict.jl")
include("machines.jl")
include("treeBasics.jl")
include("treeIterators.jl")
include("treeMeasure.jl")
include("treeTDVP.jl")
include("treeDTDVP.jl")
include("mpsBasics.jl")
include("chainTDVP.jl")
include("chain2TDVP.jl")
include("chainDMRG.jl")
include("models.jl")
include("logging.jl")
include("run_all.jl")
include("run_1TDVP.jl")
include("run_2TDVP.jl")
include("run_DTDVP.jl")
include("run_A1TDVP.jl")

include("chainA1TDVP.jl")
 
function runsim(dt, tmax, A, H;
                method=:TDVP1,
                machine=LocalMachine(),
                params=[],
                obs=[],
                convobs=[],
                convparams=error("Must specify convergence parameters"),
                save=false,
                plot=save,
                savedir=string(homedir(),"/MPSDynamics/"),
                unid=randstring(5),
                name=nothing,
                kwargs...
                )
    remote = typeof(machine) == RemoteMachine
    remote && update_machines([machine])

    if typeof(convparams) <: Vector && length(convparams) > 1
        convcheck = true
        numconv = length(convparams)
    else
        convcheck = false
        convparams = typeof(convparams) <: Vector ? only(convparams) : convparams
    end

    if save || plot
        if savedir[end] != '/'
            savedir = string(savedir,"/")
        end
        isdir(savedir) || mkdir(savedir)
        open_log(dt, tmax, convparams, method, machine, savedir, unid, name, params, obs, convobs, convcheck, kwargs...)
    end

    paramdict = Dict([[(par[1], par[2]) for par in params]...,
                      ("dt",dt),
                      ("tmax",tmax),
                      ("method",method),
                      ("convparams",convparams),
                      ("unid",unid),
                      ("name",name)
                      ]
                     )
    
    errorfile = "$(unid).e"

    tstart = now()
    A0, dat = try
        out = launch_workers(machine) do pid
            print("loading MPSDynamics............")
            if !GPU
                #@everywhere pid eval(using MPSDynamics)
            end
            println("done")
            out = fetch(@spawnat only(pid) run_all(dt, tmax, A, H;
                                                       method=method,
                                                       obs=obs,
                                                       convobs=convobs,
                                                       convparams=convparams,
                                                       kwargs...))
            return out
        end
        if typeof(out) <: Distributed.RemoteException
            throw(out)
        else
            A0, dat = out
            save && save_data(savedir, unid, convcheck, dat["data"], convcheck ? dat["convdata"] : nothing, paramdict)
            plot && save_plot(savedir, convcheck, unid, dat["data"]["times"], convcheck ? dat["convdata"] : dat["data"], convparams, convobs)
            dat = flatten(dat)
            return A0, dat
        end
    catch e
        save && error_log(savedir, unid)
        showerror(stdout, e, catch_backtrace())                
        println()
        save && open(string(savedir, unid, "/", errorfile), "w+") do io
            showerror(io, e, catch_backtrace())
        end
        return (nothing, nothing)
    finally
        telapsed = canonicalize(Dates.CompoundPeriod(now() - tstart))
        if save
            output = length(filter(x-> x!=errorfile && x!="info.txt", readdir(string(savedir, unid)))) > 0
            close_log(savedir, unid, output, telapsed)
        end
        println("total run time : $telapsed")
    end
    return A0, dat
end

export sz, sx, sy, numb, crea, anih, unitcol, unitrow, unitmat, spinSX, spinSY, spinSZ, SZ, SX, SY

export chaincoeffs_ohmic, spinbosonmpo, methylbluempo, methylbluempo_correlated, methylbluempo_correlated_nocoupling, methylbluempo_nocoupling, ibmmpo, methylblue_S1_mpo, methylbluempo2, twobathspinmpo, xyzmpo

export productstatemps, physdims, randmps, bonddims, elementmps

export measure, measurempo, OneSiteObservable, TwoSiteObservable, FockError, errorbar

export runsim, run_all

export Machine, RemoteMachine, LocalMachine, init_machines, update_machines, launch_workers, rmworkers

export randtree

export readchaincoeffs, h5read, load

export println, print, show

export @LogParams

export MPOtoVector, MPStoVector

end

