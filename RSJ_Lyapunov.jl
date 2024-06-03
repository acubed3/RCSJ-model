using DynamicalSystems
using Plots
using DifferentialEquations
using Distributions
using ProgressMeter
using DelimitedFiles

function phase_lock_areas!(omega)

	@show(omega)

	function RSJ!(dtheta, theta, p, t)
		omega, A, B = p
		dtheta[1] = -sin(theta[1]) + B + A*cos(omega*theta[2])
		dtheta[2] = 1
		nothing
	end

	A_min = -2.0;
	A_max = +8.0;
	A_step = 0.01;

	A_values = range(A_min, A_max, step=A_step) |> collect;

	B_min = -2.0;
	B_max = 2.0;
	B_step = 0.01;

	B_values = range(B_min, B_max, step=B_step) |> collect;

	theta_0 = [2*pi*rand(), 1];
	t0 = 100000;

	A_size = length(A_values);
	B_size = length(B_values);

	wm_values = zeros(B_size, A_size);

	@showprogress Threads.@threads for i=1:B_size  
		for j=1:A_size
			sys = ContinuousDynamicalSystem(RSJ!, theta_0, (omega, A_values[j], B_values[i]))
			wm_values[i,j] = ChaosTools.lyapunov(sys, t0, d0=1)
		end
	end

	name_pattern = join(["ph_Lyap_omega_", string(omega)])
	name_pattern = replace(name_pattern, "." => "_")
	name = join([name_pattern, ".csv"])

	writedlm(name,  wm_values, ',')
	
end

omega = parse(Float64, ARGS[1])
phase_lock_areas!(omega)