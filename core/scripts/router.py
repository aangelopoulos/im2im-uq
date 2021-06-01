import wandb
import yaml

if __name__ == "__main__":
  wandb.init()
  print(wandb.config)
  print("Hello, World!")
