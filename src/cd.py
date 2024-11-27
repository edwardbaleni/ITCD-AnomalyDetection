# check out: https://github.com/hfawaz/cd-diagram/blob/master/example.csv
# for how dataset must look
# check out:  https://github.com/hfawaz/cd-diagram/tree/master
# to see full critial difference code.
# ADBENCH uses this code!!

from utils import cdDiagram
df_perf = aucroc_df.iloc[:, 4:].T
df_perf.reset_index(inplace=True)
df_perf.columns = ['classifier_name', 'accuracy']
cdDiagram.draw_cd_diagram(df_perf=df_perf, title='Accuracy', labels=True)