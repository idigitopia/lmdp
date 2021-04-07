from tqdm import  tqdm
import time
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger("lmdp_logger")

def evaluate_on_env(env, policy_func, eps_count=30, verbose=False, render = False, lag = 0, progress_bar=True, every_step_hook=None, action_repeat=1, eval_eps = 0.001,
                    render_mode = "human"):
    """
    takes input environment and a policy and returns average rewards
    latent policy flag = True if the policy is a discrete policy
    :param env:
    :param policy_func:
    :param eps_count:
    :param return_info:
    :param verbose:
    :param policy_is_discrete:
    :return:
    """

    all_eps_rewards, all_eps_step_counts = [],[]
    run_info = {}
    iter__ = tqdm(range(eps_count)) if progress_bar else range(eps_count)
    action_counts = defaultdict(lambda :0)
    last_action , same_action_count = 0,0
    for e in iter__:
        eps_rewards, eps_steps, eps_infos, eps_renders = [], [], [], []
        sum_rewards, sum_steps, steps = 0,0,0
        state_c = env.reset()

        done = False
        while not done:
            eps_rewards.append(0)
            eps_steps.append(0)

            sum_steps += 1
            policyAction = policy_func(state_c) if np.random.uniform(0,1) > eval_eps else env.action_space.sample()
            # logic to not get stuck
            for _ in range(action_repeat):
                # action_counts[policyAction] += 1
                state_c, reward, done, info = env.step(np.array(policyAction))
                sum_rewards += reward
                if every_step_hook is not None:
                    every_step_hook(env, state_c)
                if(render):
                    eps_renders.append(env.render(mode = render_mode))
                    time.sleep(lag)

            eps_rewards[-1] = sum_rewards - (eps_rewards[-2] if len(eps_rewards) > 1 else 0)
            eps_steps[-1] =  sum_steps - (eps_steps[-2] if len(eps_steps) > 1 else 0)

        all_eps_rewards.append(sum_rewards)
        all_eps_step_counts.append(sum_steps)
        run_info["Run"+str(e)]= {"sum_reward":sum_rewards,
                                 "sum_steps":sum_steps,
                                 "rewards":eps_rewards,
                                 "steps":eps_steps,
                                 "renders":eps_renders,
                                 }


        if verbose:
            logger.debug(f"Episode:{e}, Steps:{steps}, Reward: {sum_rewards}")


    info = {"avg_reward":  np.mean(all_eps_rewards),
            "std_reward": np.std(all_eps_rewards),
            "avg_steps": np.mean(all_eps_step_counts),
            "std_steps" : np.std(all_eps_step_counts),
            "max_reward":  max(all_eps_rewards),
            "min_reward":  min(all_eps_rewards),
            "max_steps": max(all_eps_step_counts),
            "min_steps": min(all_eps_step_counts) ,
            # "action_counts": action_counts,
            "run_info": run_info}

    logger.info(f"Average Reward: {info['avg_reward']}")
    return info["avg_reward"], info
