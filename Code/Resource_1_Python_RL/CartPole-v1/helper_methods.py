import os
import gymnasium as gym
import matplotlib.pyplot as plt


def setup_training_directories():
    # Get the directory where the current Python script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the base directory for training artifacts relative to the script
    training_dir = os.path.join(script_dir, 'Training')
    # Define the subdirectories for logs and saved models
    log_path = os.path.join(training_dir, 'Logs')
    model_path = os.path.join(training_dir, 'SavedModels')
    # Create the directories if they don't exist
    # exist_ok=True means it won't raise an error if the directory already exists
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    return log_path, model_path



def testing_function(model, number_episode, environment,Display = False):
    result = []
    for episode in range(number_episode):
        obs = environment.reset()
        done = False
        score = 0
        Cart_Position =[]
        Cart_Velocity = []
        Pole_Angle = []
        Pole_Velocity = []
        Action_List = []
        x_axis =[]
        counter = 0

        while not done:
            environment.render()
            action = model.predict(obs)
            obs, reward, done, info = environment.step(action)
            Cart_Position.append(obs[0])
            Cart_Velocity.append(obs[1])
            Pole_Angle.append(obs[2])
            Pole_Velocity.append(obs[3])
            Action_List.append(action)
            x_axis.append(counter)
            counter +=1
            score+= reward
        print(f" The espisode number {episode} has score {score}")
        result.append([episode, score])
        fig, axis = plt.subplots(2,2)
        axis[0,0].plot(x_axis, Cart_Position)
        axis[0,0].set_title("The plot of Cart Position")

        axis[0, 1].plot(x_axis, Cart_Velocity)
        axis[0, 1].set_title("The plot of Cart Velocity")

        axis[1, 0].plot(x_axis, Pole_Angle)
        axis[1, 0].set_title("The plot of Pole Angle")

        axis[1, 1].plot(x_axis, Pole_Velocity)
        axis[1, 1].set_title("The plot of Pole Velocity")
        if Display:
            plt.show()
    return result


def testing_function_2(model, number_episode, environment, Display = False):
    all_episode_results = []
    # For CartPole, environment is a DummyVecEnv wrapping Monitor wrapping CartPole-v1
    # We'll need to access the inner environment for rendering or specific data if Display is True

    for episode in range(number_episode):
        # DummyVecEnv's reset returns (batched_obs, batched_info)
        obs = environment.reset()
        done = False
        score = 0
        Cart_Position =[]
        Cart_Velocity = []
        Pole_Angle = []
        Pole_Velocity = []
        Action_List = []
        x_axis =[]
        counter = 0

        while not done:
            if Display:
                # Access the underlying environment for rendering
                # environment.envs[0] gives the first (and only) environment in DummyVecEnv
                # .render() on that specific env will open the window
                environment.envs[0].render()

            # model.predict returns (batched_action, _states)
            action, _states = model.predict(obs, deterministic=True)

            # DummyVecEnv's step returns (batched_obs, batched_reward, batched_terminated, batched_truncated, batched_info)
            # You need to unpack 5 elements, even if you only have one environment.
            obs, reward, terminated, truncated, info = environment.step(action)

            # The 'done' condition for the while loop
            done = terminated[0] or truncated[0] # Access the boolean for the single environment

            # CartPole's observation (obs) is a numpy array: [cart_pos, cart_vel, pole_angle, pole_vel]
            # DummyVecEnv makes obs a batch, so obs[0] is the observation for the first env.
            Cart_Position.append(obs[0][0]) # obs[0] is the first env's observation array
            Cart_Velocity.append(obs[0][1])
            Pole_Angle.append(obs[0][2])
            Pole_Velocity.append(obs[0][3])

            Action_List.append(action[0]) # Action from model.predict is also batched, take the first one
            x_axis.append(counter)
            counter +=1
            score += reward[0] # Reward is also batched

        print(f"The episode number {episode} has score {score:.2f}")
        all_episode_results.append({"episode": episode, "score": score, "steps": counter})

        fig, axis = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Episode {episode} Analysis (Score: {score:.2f})", fontsize=16)

        axis[0, 0].plot(x_axis, Cart_Position)
        axis[0, 0].set_title("Cart Position")
        axis[0, 0].set_xlabel("Timestep")
        axis[0, 0].set_ylabel("Position")

        axis[0, 1].plot(x_axis, Cart_Velocity)
        axis[0, 1].set_title("Cart Velocity")
        axis[0, 1].set_xlabel("Timestep")
        axis[0, 1].set_ylabel("Velocity")

        axis[1, 0].plot(x_axis, Pole_Angle)
        axis[1, 0].set_title("Pole Angle (radians)")
        axis[1, 0].set_xlabel("Timestep")
        axis[1, 0].set_ylabel("Angle")

        axis[1, 1].plot(x_axis, Pole_Velocity)
        axis[1, 1].set_title("Pole Velocity")
        axis[1, 1].set_xlabel("Timestep")
        axis[1, 1].set_ylabel("Velocity")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if Display:
            plt.show()
        else:
            plt.close(fig) # Close figure if not displayed to free memory
            # You might want to save figures here if Display is False

    return all_episode_results


