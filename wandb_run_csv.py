import wandb

api = wandb.Api()                                                                                                                              
  # Replace with your actual entity/project/run
run = api.run("simosoftware4-other/light-dark-slac_POMDP/m5wbzowz")
df = run.history()
df.to_csv("wandb_run_history_old_working.csv")
print(df.columns.tolist())  # shows you what metrics were logged