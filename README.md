# MPSDynamics with GPU Acceleration

Tensor network code used for "Environmentally driven symmetry-breaking quenches 
dual fluorescence in proflavine" by Kye E. Hunter, Yuezhi Mao, Alex W. Chin, and
 Tim J. Zuehlsdoff. 

This fork of [MPSDynamics.jl](https://github.com/angusdunnett/MPSDynamics.git) 
has been modified by Kye E. Hunter and Evan Lambertson so that many parts 
are now GPU accelerated. Please see the upstream repository for additional 
background and documentation. The majority of the changes in this version are 
in `src/MPSDynamics.jl` and `src/tensorOps.jl`.

## Installation

The package may be installed by typing the following into a Julia REPL

```julia
] add https://github.com/tjz21/MPSDynamics_GPU.git
```

It is best to install the package with GPU acceleration while logged into a machine 
which has GPUs. This might not usually be an issue, but for computing clusters it 
is best to install the package from one of the nodes on which you intend to run 
these calculations. Otherwise, the packages will reinstantiate on the first run.

## Usage

Using this adapted version is generally similar to the upstream package, and a 
commented example of how to calculate linear response functions is included in this 
repository. The major differences in this version are:
* MPSDynamics should be loaded with the `include` function rather than a `using`
 statement. This is a result of needing to overwriting some functions/macros in
 a couple of the dependencies.
* GPU usage is assumed in this version. To use CPUs, the original MPSDynamics 
version should be used.

