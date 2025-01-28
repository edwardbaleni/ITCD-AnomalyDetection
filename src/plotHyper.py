# %%
import joblib
import optuna

study = joblib.load("results/hyperparameter/optuna_study.pkl")
# %%
study.best_params
print(study.best_trial)
optuna.visualization.plot_optimization_history(study)
# %%
optuna.visualization.plot_param_importance(study)

# %%
optuna.visualization.plot_slice(study)
# %%
# Plot the parallel coordinates plot
optuna.visualization.plot_parallel_coordinate(study)
# %%

# Save study history as a dataframe
study_history_df = study.trials_dataframe()
# study_history_df.to_csv("study_history.csv", index=False)

# %%

# Plot intermediate values
optuna.visualization.plot_intermediate_values(study)
# %%
