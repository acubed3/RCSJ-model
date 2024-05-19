using DifferentialEquations
using Distributions
using DelimitedFiles
using ProgressMeter

function phase_lock_areas!(eps)

	function RSCJ!(dtheta, theta, p, t)
		eps, omega, A, B = p
		dtheta[1] = theta[2]/eps
		dtheta[2] = -theta[2]/eps -sin(theta[1]) + B + A*cos(omega*theta[3])
		dtheta[3] = 1
		nothing
	end
	
	function winding_number!(eps, omega, A, B, t0)
		theta_0 = [2*pi*rand(), 0, 1]
		tspan = (0., t0)
		
		p = (eps, omega, A, B)
		prob = ODEProblem(RSCJ!, theta_0, tspan, p)
		sol = solve(prob)
		
		phase = last(sol.u)[1]
		wm = phase/(omega*t0)
		return wm
	end

	A_min = -2.0;
	A_max = +8.0;
	A_step = 0.01;

	A_values = range(A_min, A_max, step=A_step) |> collect;

	B_min = -2.0;
	B_max = 2.0;
	B_step = 0.01;

	B_values = range(B_min, B_max, step=B_step) |> collect;

	omega = 1.0;
	t0 = 1000;

	A_size = length(A_values);
	B_size = length(B_values);

	wm_values = zeros(B_size, A_size);

	@showprogress Threads.@threads for i=1:B_size  
		for j=1:A_size
			wm_values[i,j] = winding_number!(eps, omega, A_values[j], B_values[i], t0)
		end
	end

	name_pattern = join(["ph_lock_mass_", string(eps)])
	name_pattern = replace(name_pattern, "." => "_")
	name = join([name_pattern, ".csv"])

	writedlm(name,  wm_values, ',')
	
end