function run_1TDVP(dt, tmax, A_in, H, Dmax; obs=[], timed=false, kwargs...)
    A=A_in
    data = Dict{String,Any}()

    start_time = zero(dt)
    numsteps = round(Int, abs(tmax - start_time) / abs(dt))
    times = [start_time + i*dt for i=0:numsteps]

    @printf("Dmax : %i \n", Dmax)

    exp = measure(A, obs; t=times[1])
    for i=1:length(obs)
        push!(data, obs[i].name => reshape(exp[i], size(exp[i])..., 1))
    end

    timed && (ttdvp = Vector{Float64}(undef, numsteps))

    F=nothing
    mpsembed!(A, Dmax)
    for tstep=1:numsteps
        @printf("%i/%i, t = %.3f + %.3fim ", tstep, numsteps, real(times[tstep]), imag(times[tstep]))
        println()

        if timed
            val, t, bytes, gctime, memallocs = @timed tdvp1sweep!(dt, A, H, F, tstep; kwargs...)
            println("\t","ΔT = ", t)
            A, F = val
            ttdvp[tstep] = t
        else
            A, F = tdvp1sweep!(dt, A, H, F, tstep; kwargs...)
        end

        exp = measure(A, obs; t=times[tstep])
        for (i, ob) in enumerate(obs)
            data[ob.name] = cat(data[ob.name], exp[i]; dims=ndims(exp[i])+1)
        end
    end
    timed && push!(data, "deltat"=>ttdvp)
    push!(data, "times" => times)
    return A, data
end
