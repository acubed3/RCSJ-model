using Plots
using DifferentialEquations
using Distributions
using ProgressMeter
using DelimitedFiles

function phase_lock_areas!(omega)

	@show(omega)

	function RSJ_Mobius!(du, u, p, t)
		omega, A, B = p
		du[1] = -1/2*(conj(u[2])-u[2]*u[1]^2)+1im*(B+A*cos(omega*t))*u[1]
		du[2] = -1/2*(1-abs(u[2])^2)*conj(u[1])
		nothing
	end
	
	function lyapunov_exp!(omega, A, B, t0)
	
		u0 = [exp(1im*rand()), 0.2*rand()];
		tspan = (0., t0)
		p = (omega, A, B)


		prob = ODEProblem(RSJ_Mobius!, u0, tspan, p, abstol=1e-10, reltol=1e-10)
		sol = solve(prob)
		
		w_final = last(sol.u)[2]
		if abs(w_final) < 1.0
			ratio = (1+abs(w_final))/(1-abs(w_final))
		else
			ratio = (1+abs(w_final-1e-6))/(1-abs(w_final-1e-6))
		end 
		
		return ratio
	end
	
	t0 = 100;

	A_min = -2.0;
	A_max = +8.0;
	A_step = 0.005;

	A_values = range(A_min, A_max, step=A_step) |> collect;

	B_min = -2.0;
	B_max = 2.0;
	B_step = 0.005;

	B_values = range(B_min, B_max, step=B_step) |> collect;

	A_size = length(A_values);
	B_size = length(B_values);

	ratios = zeros(B_size, A_size);

	@showprogress Threads.@threads for i=1:B_size  
		for j=1:A_size
			ratios[i,j] = lyapunov_exp!(omega, A_values[j], B_values[i], t0)
		end
	end

	name_pattern = join(["ph_FancyLyaps_omega_", string(omega)])
	name_pattern = replace(name_pattern, "." => "_")
	name = join([name_pattern, ".csv"])

	writedlm(name,  wm_values, ',')
	
end

omega = parse(Float64, ARGS[1])
phase_lock_areas!(omega)