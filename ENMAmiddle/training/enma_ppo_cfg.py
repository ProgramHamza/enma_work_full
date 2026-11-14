from isaaclab.utils import configclass


@configclass
class PPOAlgorithmCfg:
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 3.0e-4   # constant; the training loop uses plain Adam without a scheduler
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 0.5      # value applied by the training loop (clip_grad_norm_)


@configclass
class ENMAPPORunnerCfg:
    num_steps_per_env = 200
    max_iterations = 10000
    save_interval = 100
    experiment_name = "enma_middle_snn"
    empirical_normalization = False

    policy = {
        "class_name": "ENMAPolicy",
    }

    algorithm = PPOAlgorithmCfg()