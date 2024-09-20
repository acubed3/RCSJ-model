using Plots
using LinearAlgebra
using DifferentialEquations
using Distributions
using DelimitedFiles
using Sundials

function Hill_Equation!(du, u, p, t)
    omega, A, B = p
    P1 = 2*A*omega^2*cos(omega*t)/(1+B+A*cos(omega*t))
    P2 = (1+B+A*cos(omega*t))*(-1+B+A*cos(omega*t))
    P3 = 3*A^2*omega^2*sin(omega*t)^2/(1+B+A*cos(omega*t))^2
    du[1] = u[2]
    du[2] = 1/4*(P1-P2+P3)*u[1]
    nothing
end

function compute_monodromy!(omega, A, B, t0)
    tspan = (0., t0)
    p = (omega, A, B)
    
    initials_1 = [0.0+0.0*1im,1.0+0.0*1im]
    initials_2 = [1.0+0.0*1im,0.0+0.0*1im]
    
    prob1 = ODEProblem(Hill_Equation!, initials_1, tspan, p)
    sol1 = solve(prob1)

    prob2 = ODEProblem(Hill_Equation!, initials_2, tspan, p)
    sol2 = solve(prob2)
    
    monodromy_matrix = hcat(last(sol1.u), last(sol2.u))
    trace_monodromy = tr(monodromy_matrix)
    if abs(trace_monodromy)>2
        return 1
    else
        return 0
    end
end

A_min = -4.0;
A_max = 4.0;
A_step = 0.1;

A_values = range(A_min, A_max, step=A_step) |> collect;

B_min = -4.0;
B_max = 4.0;
B_step = 0.1;

B_values = range(B_min, B_max, step=B_step) |> collect;

A_size = length(A_values);
B_size = length(B_values);
print(A_size, "\n", B_size)

tr_M = zeros(B_size, A_size);

const omega = 1.0
const t0 = 2*pi/omega;

@time Threads.@threads for i=1:B_size  
    for j=1:A_size
        tr_M[i,j] = compute_monodromy!(omega, A_values[j]+1im*0.1, B_values[i], t0)
    end
end