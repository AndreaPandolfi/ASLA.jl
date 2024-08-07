using CSV
using DataFrames
using PrettyTables

path_to_folder = "paper\\real_data_example\\"

df_GG =  DataFrame(CSV.File(path_to_folder*"summary_GG.csv"))
df_InstEval =  DataFrame(CSV.File(path_to_folder*"summary_InstEval.csv"))[vcat(1:6, 9:14), :]

cases = ["Random intercepts", "Nested effect", "Random slopes", "2 way interactions", "3 way interactions", "Everything"]; cases = vcat([[case, ""] for case in cases]...);

summary_df = DataFrame(
    Case = cases,
    Real = df_GG.Real,
    Simulated = df_GG.Simulated,
    Real1 = df_InstEval.Real,
    Simulated1 = df_InstEval.Simulated
)

open(path_to_folder*"summary.tex", "w") do f
    pretty_table(f, summary_df, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c], hlines = 3:2:11, tf = tf_latex_double) 
end