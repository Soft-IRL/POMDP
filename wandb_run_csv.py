import wandb

api = wandb.Api()                                                                                                                              
  # Replace with your actual entity/project/run
run = api.run("simosoftware4-other/light-dark-slac_POMDP/5xsdhd9w")
import pandas as pd
rows = list(run.scan_history())
df = pd.DataFrame(rows)
df.to_csv("wandb_run_history_5xsdhd9w.csv", index=False)
print(f"rows: {len(df)}")
print(df.columns.tolist())