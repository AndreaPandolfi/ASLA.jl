using JLD2, FileIO
using PrettyTables

path_to_folder = "paper/cholesky_cg_runtime/"


# ------------------ TABLE with n.iters ------------------ 
D1 = FileIO.load(path_to_folder*"cholesky_runtime1.jld2", "dict_plot1")
D2 = FileIO.load(path_to_folder*"cholesky_runtime2.jld2", "dict_plot2")
D3 = FileIO.load(path_to_folder*"cholesky_runtime3.jld2", "dict_plot3")

df = DataFrame(
    G3 = 5*D3[:Is],
    iters = D1[:iters_cg],
    iters2 = D2[:iters_cg],
    iters3 = D3[:iters_cg]
)

open(path_to_folder*"cg_iters_simulated.tex", "w") do f
    pretty_table(f, df, backend=Val(:latex), alignment=[:r, :c, :c, :c], tf = tf_latex_double) 
end