import wandb

api = wandb.Api()                                                                                                                              
  # Replace with your actual entity/project/run
run = api.run("simosoftware4-other/light-dark-slac_POMDP/acq0rzt5")
df = run.history()
df.to_csv("wandb_run_history.csv")
print(df.columns.tolist())  # shows you what metrics were logged