default(size = (800,600), reuse = true)

crea(d) = diagm(-1 => [sqrt(i) for i=1:d-1])
anih(d) = Matrix(crea(d)')
numb(d) = crea(d)*anih(d)
"""
    disp(d)

Mass and frequency-weighted displacement operator 
``
X = \\frac{1}{2}(a + a^{\\dagger})  
``

"""
disp(d) = (1/sqrt(2))*(crea(d)+anih(d))
"""
    disp(d,ωvib,m)

Displacement operator 
``
X = \\frac{\\sqrt{2}}{2\\sqrt{m \\omega_{vib}}}(a + a^{\\dagger})  
``
"""
disp(d,ωvib,m) = (1/(2*sqrt(m*ωvib/2)))*(crea(d)+anih(d))
mome(d) = (1/sqrt(2))*(im*(crea(d)-anih(d)))
sx = [0. 1.; 1. 0.]
sz = [1. 0.; 0. -1.]
sy = [0. -im; im 0.]
sp = [0. 1.; 0. 0.]
sm = Matrix(sp')

Adagup = diagm(-1 => [1,0,1])
Adagdn = diagm(-2 => [1,1])
Aup = diagm(1 => [1,0,1])
Adn = diagm(2 => [1,1])
Ntot = diagm(0 => [0,1,1,2])
Nup = diagm(0 => [0,1,0,1])
Ndn = diagm(0 => [0,0,1,1])
parity = diagm(0 => [1,-1,-1,1])

unitvec(n, d) = [fill(0.0, n-1)..., 1.0, fill(0.0, d-n)...]
unitmat(d1, d2) = [i1==i2 ? 1.0 : 0.0 for i1=1:d1, i2=1:d2]
unitmat(d) = unitmat(d, d)

function unitcol(n, d)
    z = zeros(d, 1)
    z[n] = 1
    return z
end

function unitrow(n, d)
    z = zeros(1, d)
    z[n] = 1
    return z
end

abstract type LightCone end

struct siteops
    h::Array{ComplexF64, 2}
    r::Array{ComplexF64, 2}
    l::Array{ComplexF64, 2}
end

mutable struct envops
    H::Array{ComplexF64, 2}
    op::Array{ComplexF64, 2}
    opd::Array{ComplexF64, 2}
end

mutable struct obbsite
    A::Array{ComplexF64, 3}
    V::Array{ComplexF64, 2}
    obbsite(A,V) = size(A,2) == size(V,2) ? new(A,V) : error("Dimension mismatch")
end

#chose n elements of x without replacement
function choose(x, n)
    len = length(x)
    if n > len
        throw(ArgumentError("cannot chose more elements than are contained in the list"))
        return
    end
    remaining = x
    res = Vector{eltype(x)}(undef, n)
    for i in 1:n
        rnd = rand(remaining)
        res[i] = rnd
        filter!(p -> p != rnd, remaining)
    end
    return res
end

"""
     therHam(psi, site1, site2)

Calculates Hβ such that ρ = e^(-βH) for some density matrix ρ obatined from tracing out everything outside the range [site1,site2] in the MPS psi
"""
function therHam(psi, site1, site2)
    pmat = ptracemps(psi, site1, site2)
    pmat = 0.5 * (pmat + pmat')
    Hb = -log(pmat)
    S = eigen(Hb).values
    return Hb, S
end

function loaddat(dir, unid, var)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"dat_",unid,".jld"), var)
end

function loaddat(dir, unid)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"dat_",unid,".jld"))
end
function loadconv(dir, unid, var)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"convdat_",unid,".jld"), var)
end

function loadconv(dir, unid)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    load(string(dir,"convdat_",unid,".jld"))
end

function savecsv(dir, fname, dat)
    if dir[end] != '/'
        dir = string(dir,"/")
    end
    writedlm(string(dir, fname,".csv"), dat, ',')
end

"""
    eigenchain(cparams; nummodes=nothing)

"""
function eigenchain(cparams; nummodes=nothing)
    if nummodes==nothing
        nummodes = length(cparams[1])
    end
    es=cparams[1][1:nummodes]
    ts=cparams[2][1:nummodes-1]
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    return eigen(hmat)
end

"""
    thermaloccupations(β, cparams...)

"""
function thermaloccupations(β, cparams...)
    es=cparams[1]
    ts=cparams[2]
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    F = eigen(hmat)
    U=F.vectors
    S=F.values
    ((U)^2) * (1 ./ (exp.(β.*S).-1))
end

"""
    measuremodes(A, chainsection::Tuple{Int64,Int64}, e::Array{Float64,1}, t::Array{Float64,1})

"""
function measuremodes(A, chainsection::Tuple{Int64,Int64}, e::Array{Float64,1}, t::Array{Float64,1})
    N=abs(chainsection[1]-chainsection[2])+1
    e=e[1:N]
    t=t[1:N-1]
    d=size(A[chainsection[1]])[2]#assumes constant number of Fock states
    bd=crea(d)
    b=anih(d)
    hmat = diagm(0=>e, 1=>t, -1=>t)
    F = eigen(hmat)
    U = F.vectors
    return real.(diag(U' * measure2siteoperator(A, bd, b, chainsection, conj=true, herm_cis=true) * U))
end

"""
    measuremodes(A, chainsection::Tuple{Int64,Int64}, U::AbstractArray)

for longer chains it can be worth calculating U in advance


"""
function measuremodes(A, chainsection::Tuple{Int64,Int64}, U::AbstractArray)
    d=size(A[chainsection[1]])[2]#assumes constant number of Fock states
    bd=crea(d)
    b=anih(d)
    return real.(diag(U' * measure2siteoperator(A, bd, b, chainsection, conj=true, herm_cis=true) * U))
end

"""
    measuremodes(adaga, e=1.0, t=1.0)


"""
function measuremodes(adaga, e=1.0, t=1.0)
    N = size(adaga)[1]
    hmat = diagm(0=>fill(e,N), 1=>fill(t,N-1), -1=>fill(t,N-1))
    F = eigen(hmat)
    U = F.vectors
    return real.(diag(U' * adaga * U))
end

"""
    measuremodes(adaga, e::Vector, t::Vector)


"""
function measuremodes(adaga, e::Vector, t::Vector)
    N = size(adaga)[1]
    hmat = diagm(0=>e[1:N], 1=>t[1:N-1], -1=>t[1:N-1])
    F = eigen(hmat)
    U = F.vectors
    return real.(diag(U' * adaga * U))
end

"""
    measurecorrs(oper, , e::Vector, t::Vector)

### Parameters

`oper`: Square matrix (Matrix{Float64}) representing the operator to be transformed.
`e`: Vector (Vector{Float64}) of diagonal (on-site energy) chain coefficients.
`t`: Vector (Vector{Float64}) of off-diagonal (hopping terms) chain coefficients.

### Returns

Matrix{Float64}: This matrix is the operator `oper` transformed back from the chain 
representation to the representation corresponding to the extended bath. The resulting 
operator represents quantities like mode occupations or other properties in the basis 
of environmental modes associated with specific frequencies ``\\omega_i``.

### Description
    
This function performs a basis transformation of the operator `oper`. Specifically, 
this transformation reverses the unitary transformation that maps the extended bath
Hamiltonian into the chain representation. 

"""
function measurecorrs(oper, e::Vector, t::Vector)
    N = size(oper)[1]
    hmat = diagm(0=>e[1:N], 1=>t[1:N-1], -1=>t[1:N-1])
    F = eigen(hmat)
    U = F.vectors
    return (U' * oper * U)
end


"""
    cosineh(omega, bet)

Calculates the hyperbolic cosine function function based on the input parameters, 
for the Bogoliubov transformation necessary for the thermofield transformation.

# Arguments
- `omega::Float64`: The frequency parameter.
- `bet::Float64`: The beta parameter.

# Returns
- `Float64`: The result of the modified cosine function.
"""
function cosineh(omega, bet)
    return 1/sqrt(1 - exp(-omega * (bet)))
end

"""
    sineh(omega, bet)

Calculates the hyperbolic sine function function based on the input parameters, 
for the Bogoliubov transformation necessary for the thermofield transformation.

# Arguments
- `omega::Float64`: The frequency parameter.
- `bet::Float64`: The beta parameter.

# Returns
- `Float64`: The result of the modified cosine function.
"""
function sineh(omega, bet)
    return 1/sqrt(-1 + exp(omega * float(bet)))
end

"""
    physical_occup(corr_constr, corr_destr, omega, occup, b, M)

Calculates the physical occupation based on correlation matrices, omega values, 
and other parameters. The physical occupation in the original frequency environment
is computed by reverting the thermofield transformation.

# Arguments
- `corr_constr::Matrix{ComplexF64}`: The correlation construction matrix.
- `corr_destr::Matrix{ComplexF64}`: The correlation destruction matrix.
- `omega::Vector{Float64}`: The omega values.
- `occup::Matrix{Float64}`: The occupation matrix.
- `b::Float64`: The beta parameter.
- `M::Int`: The number of points for interpolation.

# Returns
- `Vector{Float64}`: The physical occupation values.
"""
function physical_occup(corr_constr, corr_destr, omega, occup, b, M)
    x = range(-1, stop=1, length=M)

    # Ensure occup is a vector
    occup_vector = vec(occup)

    occup_interp = LinearInterpolation(omega, occup_vector, extrapolation_bc=Line())
    corr_constr_interp = interpolate((omega, omega), abs.(corr_constr), Gridded(Linear()))
    corr_destr_interp = interpolate((omega, omega), abs.(corr_destr), Gridded(Linear()))

    Mhalf = div(M, 2)
    phys_occ = []

    omega_min = minimum(omega)
    omega_max = maximum(omega)
    x_rescaled = (x .+ 1) .* (omega_max - omega_min) / 2 .+ omega_min

    for el in 1:Mhalf
        ipos = Mhalf + el
        ineg = Mhalf - el + 1
        occ = (cosineh(x_rescaled[ipos], b) * sineh(x_rescaled[ipos], b) *
               (corr_destr_interp(x_rescaled[ineg], x_rescaled[ipos]) + corr_constr_interp(x_rescaled[ipos], x_rescaled[ineg])) +
               cosineh(x_rescaled[ipos], b)^2 * occup_interp(x_rescaled[ipos]) +
               sineh(x_rescaled[ipos], b)^2 * (1 + occup_interp(x_rescaled[ineg])))
        push!(phys_occ, occ)
    end

    return phys_occ
end


"""
    findchainlength(T, cparams::Vector; eps=10^-6, verbose=false)

Estimate length of chain required for a particular set of chain parameters by calculating how long an excitation on the
first site takes to reach the end. The chain length is given as the length required for the excitation to have just
reached the last site after time T. The initial number of sites in cparams has to be larger than the findchainlength result.

"""
function findchainlength(T, cparams::Vector; eps=10^-4, verbose=false)
    Nmax = length(cparams[1])
    occprev = endsiteocc(T, [cparams[1][1:Nmax], cparams[2][1:Nmax-1]])
    occ = endsiteocc(T, [cparams[1][1:Nmax-1], cparams[2][1:Nmax-2]])

    verbose && println("truncating chain...")
    verbose && println("starting chain length $Nmax")

    if abs(occ-occprev) > eps
        throw(error("Suitable truncation not found, try larger starting chain length"))
    end
    occprev=occ
    for ntrunc=Nmax-2:-1:1
        verbose && println("ntrunc = $ntrunc")
        occ = endsiteocc(T, [cparams[1][1:ntrunc], cparams[2][1:ntrunc-1]])
        if abs(occ-occprev) > eps
            return ntrunc
        end
        occprev=occ
    end
end

"""
    findchainlength(T, ωc::Float64, β=Inf)

Estimate length of chain using universal asymptotic properties of chain mapped environments given the simulation time T, the bath cut-off frequency ωc, and the inverse temperature β.
"""
function findchainlength(T, ωc::Float64, β=Inf)
    if β==Inf
        N = ceil(0.25*ωc*T + 0.5)
    else
        N = ceil(0.5*ωc*T + 0.5)
    end
    return N
end

"""
    chainprop(t, cparams)

Propagate an excitation placed initially on the first site of a tight-binding chain with parameters given by cparams for a time t and return occupation expectation for each site.

"""
function chainprop(t, cparams)
    es=cparams[1]
    ts=cparams[2]
    N=length(es)
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    F = eigen(hmat)
    U = F.vectors
    S = F.values
    [real.(transpose(U[:,1].*exp.(im*t.*S))*U[:,i]*transpose(U[:,i])*(U[:,1].*exp.(-im*t.*S))) for i in 1:N]
end

"""
    endsiteocc(t, cparams)


"""
function endsiteocc(t, cparams)
    es=cparams[1]
    ts=cparams[2]
    N=length(es)
    hmat = diagm(0=>es, 1=>ts, -1=>ts)
    F = eigen(hmat)
    U = F.vectors
    S = F.values
    real.(transpose(U[:,1].*exp.(im*t.*S))*U[:,N]*transpose(U[:,N])*(U[:,1].*exp.(-im*t.*S)))
end

function booltostr(b)
    return b ? "true" : "false"
end

function mean(nums::Vector{T}) where T <: Number
    n = length(nums)
    return sum(nums)/n
end
mean(nums...) = mean([nums...])

function var(nums::Vector{T}) where T <: Number
    n = length(nums)
    m = mean(nums)
    return sum((nums .- m) .^ 2)/n
end
var(nums...) = var([nums...])

function sd(nums::Vector{T}) where T <: Number
    return sqrt(var(nums))
end
sd(nums...) = sd([nums...])

"""
    rmsd(ob1, ob2)

Calculate the root mean squared difference between two measurements of an observable over the same time period.

"""
function rmsd(ob1, ob2)
    len = length(ob1)
    if len != length(ob2)
        throw(ArgumentError("inputs must have same length"))
    end
    return sqrt(sum((ob1 - ob2).^2)/len)
end

"""
    dynamap(ps1,ps2,ps3,ps4)

Calculate complete dynamical map to time step at which ps1, ps2, ps3 and ps4 are specified.

# Arguments
- `ps1` : time evolved system density matrix starting from initial state up
- `ps2` : time evolved system density matrix starting from initial state down
- `ps3` : time evolved system density matrix starting from initial state (up + down)/sqrt(2)
- `ps4` : time evolved system density matrix starting from initial state (up - i*down)/sqrt(2)
"""
function dynamap(ps1,ps2,ps3,ps4)
    ϵ = [
        ps1[1]-real(ps1[3])-imag(ps1[3]) ps2[1]-real(ps2[3])-imag(ps2[3]) ps3[1]-real(ps3[3])-imag(ps3[3]) ps4[1]-real(ps4[3])-imag(ps4[3]);
        ps1[4]-real(ps1[3])-imag(ps1[3]) ps2[4]-real(ps2[3])-imag(ps2[3]) ps3[4]-real(ps3[3])-imag(ps3[3]) ps4[4]-real(ps4[3])-imag(ps4[3]);
        2*real(ps1[3]) 2*real(ps2[3]) 2*real(ps3[3]) 2*real(ps4[3]);
        2*imag(ps1[3]) 2*imag(ps2[3]) 2*imag(ps3[3]) 2*imag(ps4[3])
    ]
    U = [1 (im-1)/2 -(im+1)/2 0; 0 (im-1)/2 -(im+1)/2 1; 0 1 1 0 ; 0 im -im 0]
    invU = inv(U)
    return invU*ϵ*U
end

function ttm2_calc(ps1, ps2, ps3, ps4)

    numsteps = length(ps1)
    
    ϵ = [dynamap(ps1[i],ps2[i],ps3[i],ps4[i]) for i=2:numsteps]

    T=[]

    for n in 1:numsteps-1
        dif=zero(ϵ[1])
        for m in 1:length(T)
            dif += T[n-m]*ϵ[m]
        end
        push!(T,ϵ[n] - dif)
    end
    return T
end

function ttm2_evolve(numsteps, T, ps0)

    K = length(T)
    ps = [reshape(ps0, 4, 1)]
    for i in 1:numsteps
        psnew = sum(T[1:min(i,K)] .* ps[end:-1:end-min(i,K)+1])
        ps = push!(ps, psnew)
    end
    return reshape.(ps, 2, 2)
end

"""
    entropy(rho)


"""
function entropy(rho)
    λ = eigen(rho).values
    return real(sum(map(x-> x==0 ? 0 : -x*log(x), λ)))
end

function paramstring(x, sf)
    x = round(x, digits=sf)
    xstr = string(Int64(round(x*10^sf)))
    while length(xstr) < 2*sf
        xstr = string("0",xstr)
    end
    xstr
end

function writeprint(f::IO, str...)
    print(string(str...))
    write(f, string(str...))
end
function writeprintln(f::IO, str...)
    println(string(str...))
    write(f, string(str...,"\n"))
end
function writeprint(f::Vector{T}, str...) where T <: IO
    print(string(str...))
    write.(f, string(str...))
end
function writeprintln(f::Vector{T}, str...) where T <: IO
    println(string(str...))
    write.(f, string(str...,"\n"))
end

"""
    MPOtoVector(mpo::MPO)

Convert an ITensors chain MPO into a form compatible with MPSDynamics

"""
function MPOtoVector(mpo::MPO)
    N = length(mpo)
    H = [Array(mpo[i], inds(mpo[i])...) for i=1:N]
    dims=size(H[1])
    H[1] = reshape(H[1], 1, dims...)
    dims=size(H[N])
    H[N] = reshape(H[N], dims[1], 1, dims[2], dims[3])
    return H
end
