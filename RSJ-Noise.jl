using DifferentialEquations
using Distributions
using DelimitedFiles
using ProgressMeter

function phase_lock_areas!(D, omega, A_max, A_min, B_max, B_min, A_step, B_step, t0)
	# define RSJ model as system of two 1st order ODE
	function RSJ!(dtheta, theta, p, t)
		omega, A, B, D = p
		dtheta[1] = -sin(theta[1]) + B + A*cos(omega*theta[2])
		dtheta[2] = 1
		nothing
	end
	# define white noise term
	function noise!(dtheta, theta, p, t)
		omega, A, B, D = p
		dtheta[1] = sqrt(2*D)
		dtheta[2]
		nothing
	end
	# winding number computation
	function winding_number!(omega, A, B, D, t0)
		theta_0 = [2*pi*rand(),1]
		tspan = (0., t0)
		
		p = (omega, A, B, D)
		prob = SDEProblem(RSJ!, noise!, theta_0, tspan, p)
		sol = solve(prob)
		
		phase = last(sol.u)[1]
		wm = phase/(omega*t0)
		return wm
	end
	# create grid
	A_values = range(A_min, A_max, step=A_step) |> collect;
	B_values = range(B_min, B_max, step=B_step) |> collect;
	# chekc grid size
	A_size = length(A_values);
	B_size = length(B_values);
	# create zero array of shape as of grid
	wm_values = zeros(B_size, A_size);
	# compute winding number for each point on 2D grid
	@showprogress Threads.@threads for i=1:B_size  
		for j=1:A_size
			wm_values[i,j] = winding_number!(omega, A_values[j], B_values[i], D, t0)
		end
	end
	# export winding numbers 2D array into csv file
	name_pattern = join(["ph_lock_D_", string(D), "_Amax", string(A_max), "_Bmax_", B_max, "_dA_", A_step, "_dB", B_step, "_w_", omega])
	name_pattern = replace(name_pattern, "." => "_")
	name = join([name_pattern, ".csv"])

	writedlm(name,  wm_values, ',')
	
end