# %%
import joblib
import optuna
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

# %%
from os import listdir
path = "results/hyperparameter/"
home = listdir(path)

abod_files = [f'{path}{file}' for file in home if "ABOD" in file]
lof_files = [f'{path}{file}' for file in home if "LOF" in file]
pca_files = [f'{path}{file}' for file in home if "PCA" in file]
eif_files = [f'{path}{file}' for file in home if "EIF" in file]


# %%

# ABOD
study = [joblib.load(file) for file in abod_files]

# Plot optimization history for all studies on the same plot
plt.figure(figsize=(10, 6))
for i, s in enumerate(study):
    df = s.trials_dataframe()
    df = df.sort_values(by='params_n_neighbors')
    plt.plot(df['params_n_neighbors'], df['value'], label=f'Orchard {i+1}')
plt.xlabel('Trial')
plt.ylabel('Objective Value')
plt.title('Optimization History')
plt.legend()
plt.show()

study[0].best_params
print(study[0].best_trial)
optuna.visualization.plot_optimization_history(study[0])


optuna.visualization.plot_slice(study[19])
# Plot the parallel coordinates plot
optuna.visualization.plot_parallel_coordinate(study[19])

# Save study history as a dataframe
study_history_df = study[0].trials_dataframe()
# study_history_df.to_csv("study_history.csv", index=False)

# Plot intermediate values
optuna.visualization.plot_intermediate_values(study[0])

# Plot objective value against number of neighbors
# for i in range(30):
#     study_history_df = study[i].trials_dataframe()
#     plt.figure(figsize=(10, 6))
#     plt.scatter(study_history_df['params_n_neighbors'], study_history_df['value'])
#     plt.xlabel('Number of Neighbors')
#     plt.ylabel('Objective Value')
#     plt.title(f'Objective Value vs Number of Neighbors for Orchard {i+1}')
#     plt.show()

# TODO:
# either just demonstrate number

# method is unstable


# %% TODO: LOF



# %% TODO: PCA

study = [joblib.load(file) for file in pca_files]

# Plot optimization history for all studies on the same plot
plt.figure(figsize=(10, 6))
for i, s in enumerate(study):
    df = s.trials_dataframe()
    plt.plot(df['number'], df['value'], label=f'Study {i+1}')
plt.xlabel('Trial')
plt.ylabel('Objective Value')
plt.title('Optimization History')
plt.legend()
plt.show()

study[0].best_params
print(study[0].best_trial)
optuna.visualization.plot_optimization_history(study[0])


optuna.visualization.plot_slice(study[0])
# Plot the parallel coordinates plot
optuna.visualization.plot_parallel_coordinate(study[0])

# Save study history as a dataframe
study_history_df = study[0].trials_dataframe()


for i in range(20):
    study_history_df = study[i].trials_dataframe()
    plt.figure(figsize=(20, 12))
    plt.scatter(study_history_df['params_n_selected_components'], study_history_df['value'], s=300)
    plt.xlabel('Number of Components', fontsize=30)
    plt.ylabel('Average Precision', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(f"results/hyperparameter_selection/PCA/PCA_Select_{i+1}.png")
    # plt.title(f'Objective Value vs Number of Neighbors for Orchard {i+1}')
    plt.show()

for i in range(30):
    print(study[i].best_params)


plt.figure(figsize=(20, 12))
for i, s in enumerate(study[0:5]):
    df = s.trials_dataframe()
    df = df.sort_values(by='params_n_selected_components')
    plt.plot(df['params_n_selected_components'], df['value'], label=f'Study {i+1}', linewidth=5)
plt.xlabel('Number of Components', fontsize=30)
plt.ylabel('Average Precision', fontsize=30)
# plt.title('Optimization History', fontsize=16)
plt.legend(fontsize=25, loc='upper right')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.savefig("results/hyperparameter_selection/PCA/PCA_Select.png")
plt.show()












# %% TODO: EIF
# this is the only one that benefits from parameter importance
study = [joblib.load(file) for file in eif_files]

# Plot optimization history for all studies on the same plot
plt.figure(figsize=(10, 6))
for i, s in enumerate(study):
    df = s.trials_dataframe()
    plt.plot(df['number'], df['value'], label=f'Study {i+1}')
plt.xlabel('Trial')
plt.ylabel('Objective Value')
plt.title('Optimization History')
plt.legend()
plt.show()

study[0].best_params
print(study[0].best_trial)
optuna.visualization.plot_optimization_history(study[4])


optuna.visualization.plot_slice(study[27])
# Plot the parallel coordinates plot
optuna.visualization.plot_parallel_coordinate(study[5])

# Save study history as a dataframe
study_history_df = study[0].trials_dataframe()
# study_history_df.to_csv("study_history.csv", index=False)

# Plot intermediate values
optuna.visualization.plot_intermediate_values(study[0])

for i in range(5):
    fig = optuna.visualization.plot_param_importances(study[i])
    fig.update_layout(title_text='', font=dict(size=40))
    fig.update_layout(title_font=dict(size=50, color='black', family='bold'))
    fig.write_html(f"results/hyperparameter_selection/EIF/Importance/Orchard_{i+1}.html")
    fig.show()



# %%

data = joblib.load("results/training/data0_40.pkl")


full_data = pd.concat(data)


# proposed 
# Need to chunk images (block cross-val)

for i in range(20):
    study_history_df = study[i].trials_dataframe()
    plt.figure(figsize=(20, 12))
    plt.scatter(study_history_df['params_extension_level'], study_history_df['value'], s=300)
    plt.xlabel('Number of Components', fontsize=30)
    plt.ylabel('Average Precision', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # plt.savefig(f"results/hyperparameter_selection/PCA/PCA_Select_{i+1}.png")
    # plt.title(f'Objective Value vs Number of Neighbors for Orchard {i+1}')
    plt.show()
# %%
