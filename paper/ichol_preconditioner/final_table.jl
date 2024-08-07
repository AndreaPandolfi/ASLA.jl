using CSV
using DataFrames
using PrettyTables

path_to_folder = "paper\\ichol_preconditioner\\"

df_GG =  DataFrame(CSV.File(path_to_folder*"summary_GG.csv"))
df_IE =  DataFrame(CSV.File(path_to_folder*"summary_IE.csv"))#[vcat(1:3, 5:end), :]

cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"];# cases = vcat([[case, ""] for case in cases]...);

summary_df = DataFrame(
    Case = cases,
    GGJac = df_GG.Real_jac,
    GGIchol = df_GG.Real_ichol,
    IEJac = df_IE.Real_jac,
    IEIchol = df_IE.Real_ichol
)

open(path_to_folder*"summary_ichol.tex", "w") do f
    pretty_table(f, summary_df, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c], hlines = 1:7, tf = tf_latex_double) 
end