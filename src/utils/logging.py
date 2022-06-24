from collections import defaultdict
import logging
import wandb
import numpy as np
import torch as th

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_wandb = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_wandb(self, args):
        self.use_wandb = True
        if isinstance(args.wandb_project, str):
            project_name = args.wandb_project
        else:
            project_name = args.env + "-" + args.env_args["map_name"]
        if isinstance(args.wandb, str):
            group_name = args.wandb
        else:
            group_name = args.name
        config = args.__dict__.copy()
        config.update(config["env_args"])
        del config["env_args"]
        wandb.init(
            entity=args.wandb_user,
            project=project_name,
            group=group_name,
            config=config,
        )

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_wandb:
            wandb.log({key: value}, step=t, commit=False)

    def log_video(self, video, t, name="visualisation"):
        if self.use_wandb:
            vid_seq = np.transpose(video, (0, 3, 1, 2)) # from (time x height x width x channel) to (time x channel x height x width)
            wandb_video = wandb.Video(vid_seq, fps=30, format="mp4")
            wandb.log({name: wandb_video}, step=t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(th.mean(th.tensor([float(x[1]) for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)
        # Reset stats to avoid accumulating logs in memory
        self.stats = defaultdict(lambda: [])


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

