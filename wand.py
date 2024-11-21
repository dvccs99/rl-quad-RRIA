import wandb

# Initialize a new run
wandb.init(project="my_project")

# Log some metrics
for i in range(10):
    wandb.log({"metric": i})