module MPSDynamics

using CUDA
using cuTENSOR
using JLD, HDF5, Random, Dates, Plots, Printf, Distributed, LinearAlgebra, DelimitedFiles, KrylovKit, GraphRecipes, SpecialFunctions, Interpolations, TensorOperations, ITensors, ITensorMPS

function Base.getindex(Q::LinearAlgebra.LQPackedQ{<:Any, <:CuArray}, ::Colon, j::Int)
  y = CuArray{eltype(Q)}(diagm(size(Q, 2), 1, (-j + 1) => [1]))
  return lmul!(Q, y)
end  
function Base.getindex(Q::LinearAlgebra.LQPackedQ{<:Any, <:CuArray}, i::Int, j::Int)
  x = CuArray{eltype(Q)}(diagm(size(Q, 2), 1, (-j + 1) => [1]))
  y = CuArray{eltype(Q)}(diagm(size(Q, 2), 1, (-i + 1) => [1]))
  Q = lmul!(Q, y)
  return sum(Q * x)
end
function CuMatrix{T}(Q::LinearAlgebra.LQPackedQ{S}) where {T,S}
  return CuMatrix{T}(lmul!(Q, CuMatrix{S}(I, size(Q, 1), min(size(Q.factors)...))))
end

# Define CUDA versions of LAPACK lq functions in terms of qr
function LinearAlgebra.LAPACK.orglq!(A::CuArray, tau::CuArray)
  return CuArray(transpose(CUDA.CUSOLVER.orgqr!(CuArray(transpose(A)), tau ))) 
end
function LinearAlgebra.LAPACK.ormlq!(side::Char, trans::Char, A::CuArray, tau::CuArray, C::CuArray)
    return CuArray(transpose(CUDA.CUSOLVER.ormqr!(side, trans, CuArray(transpose(A)), tau, CuArray(C))))
end

function LinearAlgebra.lq!(A::CuArray{T, 2}) where T
  Q, R = qr(CuArray(transpose(A)))
  if typeof(Q) <: LinearAlgebra.QRCompactWYQ
    Q_factors, Q_t = Q.factors, Q.T
  elseif typeof(Q) <: LinearAlgebra.QRPackedQ
    Q_factors, Q_t = Q.factors, Q.τ
  else
    println("don't know what type Q is")
  end
  Q = LinearAlgebra.LQPackedQ(CuMatrix(transpose(Q_factors)), CuMatrix(transpose(Q_t))[1:end])
  L = CuArray(transpose(R))
  return L, Q
end

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
include("mpsBasics.jl")
include("chainTDVP.jl")
include("models.jl")
include("logging.jl")
include("run_all.jl")
include("run_1TDVP.jl")
include("utilities.jl")

"""
    runsim(dt, tmax, A, H; 
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

Propagate the MPS `A` with the MPO `H` up to time `tmax` in time steps of `dt`. The final MPS is returned to `A` and the measurement data is returned to `dat` 

# Arguments

* `method`: Several methods are implemented in MPSDynamics. `:TDVP1` refers to 1-site TDVP on tree and chain MPS, `:TDVP2` refers to 2-site TDVP on chain MPS, `:DTDVP` refers to a variant of 1-site TDVP with dynamics bond-dimensions on chain MPS
* `machine`: `LocalMachine()` points local ressources, `RemoteMachine()` points distant ressources
* `params`: list of parameters written in the log.txt file to describe the dynamics. Can be listed with @LogParams(). 
* `obs`: list of observables that will be measured at every time step for the most accurate convergence parameter supplied to `convparams` 
* `convobs`: list of observables that will be measure at every time step for every convergence parameter supplied to `convparams` 
* `convparams`: list of convergence parameter with which the propagation will be calculated for every parameter. At each parameter, `convobs` are measured while `obs` are measured only for the most accurate dynamics
* `save`: Used to choose whether the data will also be saved to a file  
* `plot`: Used to choose whether plots for 1D observables will be automatically generated and saved along with the data
* `savedir`: Used to specify the path where resulting files are stored
* `unid`: Used to specify the name of the directory containing the resulting files
* `name`: Used to describe the calculation. This name will appear in the log.txt file
 
"""
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

    if save || plot || (:onthefly in keys(kwargs) && !isempty(kwargs[:onthefly][:save_obs]) && kwargs[:onthefly][:savedir] == "auto")
        if savedir[end] != '/'
            savedir = string(savedir,"/")
        end
        isdir(savedir) || mkdir(savedir)
        open_log(dt, tmax, convparams, method, machine, savedir, unid, name, params, obs, convobs, convcheck, kwargs...)
        if :onthefly in keys(kwargs)
            mkdir(string(savedir, unid, "/tmp/"))
            kwargs[:onthefly][:savedir] = string(savedir, unid, "/tmp/")
        end
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
            #@everywhere pid eval(using MPSDynamics)
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

export numb, crea, anih, unitcol, unitrow, unitmat

export spinbosonmpo, methylbluempo2

export productstatemps, physdims, randmps, bonddims, elementmps

export measure, measurempo, OneSiteObservable, TwoSiteObservable, RhoReduced, FockError, errorbar

export runsim, run_all

export Machine, RemoteMachine, LocalMachine, init_machines, update_machines, launch_workers, rmworkers

export randtree

export readchaincoeffs, h5read, load, findchainlength

export println, print, show

export @LogParams

export MPOtoVector

export rhoreduced_2sites, rhoreduced_1site

export onthefly, mergetmp

end

