using CSV
using DataFrames
using PrettyTables

path_to_folder = joinpath("paper", "real_data_example")

### CG ITERS
df_GG =  DataFrame(CSV.File(joinpath(path_to_folder,"summary_GG.csv")))
df_InstEval =  DataFrame(CSV.File(joinpath(path_to_folder, "summary_InstEval.csv")))[vcat(1:6, 9:14), :]

cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]; cases = vcat([[case, ""] for case in cases]...);

summary_df = DataFrame(
    Case = cases,
    Real = df_GG.Real,
    Simulated = df_GG.Simulated,
    Real1 = df_InstEval.Real,
    Simulated1 = df_InstEval.Simulated
)

open(joinpath(path_to_folder, "summary.tex"), "w") do f
    pretty_table(f, summary_df, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c], hlines = 3:2:11, tf = tf_latex_double) 
end

### FLOPS
df_GG =  DataFrame(CSV.File(joinpath(path_to_folder, "cost_GG.csv")))
df_InstEval =  DataFrame(CSV.File(joinpath(path_to_folder, "cost_InstEval.csv")))[vcat(1:6, 9:14), :]

cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]; cases = vcat([[case, ""] for case in cases]...);

flops_df = DataFrame(
    Case = cases,
    Real = df_GG.Real,
    Simulated = df_GG.Simulated,
    Real1 = df_InstEval.Real,
    Simulated1 = df_InstEval.Simulated
)

open(joinpath(path_to_folder, "flops_summary.tex"), "w") do f
    pretty_table(f, flops_df, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c], hlines = 3:2:11, tf = tf_latex_double) 
end


### RUNNING TIMES
using FileIO, JLD2
df_GG = load(joinpath(path_to_folder, "time_GG.jld2"), "time_df")
df_InstEval =  load(joinpath(path_to_folder, "time_IE.jld2"), "time_df")[vcat(1:6, 9:14), :]

using Printf
round(x) = @sprintf("%.2f", x)

function convert_time_str(val::Real)
    if val < .1
        # Convert to ms
        val_ms = val * 1000
        # Replace only the first number in the string
        return return @sprintf("%.2f ms", val_ms)
    else
        # Keep in seconds
        return return @sprintf("%.2f s", val)
    end
end

get_string_time(ratio::Real, t_cg::Real) = "$(round(ratio)) ($(convert_time_str(t_cg)))"


cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]; 
cases = vcat([[case, ""] for case in cases]...);

time_df = DataFrame(
    Case = cases,
    Real = [get_string_time(ratio, t_cg) for (ratio, t_cg) in df_GG.Real],
    Simulated = [get_string_time(ratio, t_cg) for (ratio, t_cg) in df_GG.Simulated],
    Real1 = [get_string_time(ratio, t_cg) for (ratio, t_cg) in df_InstEval.Real],
    Simulated1 = [get_string_time(ratio, t_cg) for (ratio, t_cg) in df_InstEval.Simulated]
)

open(joinpath(path_to_folder, "time_summary.tex"), "w") do f
    pretty_table(f, time_df, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c], hlines = 3:2:11, tf = tf_latex_double) 
end
