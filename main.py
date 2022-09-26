
##ä############################################# Simple test of visualize
# env = Environment()
# env.reset()
# print(env.visible_state)

# done = False
# while not done:
#     _, _, done = env.step(2) # Down

# animate(env.history)


################################################# Simple train

gamma = 0.9 # Discounted reward factor

mse = nn.MSELoss()

def finish_episode(e, actions, values, rewards):
    
    # Calculate discounted rewards, going backwards from end
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.Tensor(discounted_rewards)

    # Use REINFORCE on chosen actions and associated discounted rewards
    value_loss = 0
    for action, value, reward in zip(actions, values, discounted_rewards):
        reward_diff = reward - value.data[0] # Treat critic value as baseline
        action.reinforce(reward_diff) # Try to perform better than baseline
        value_loss += mse(value, Variable(torch.Tensor([reward]))) # Compare with actual reward

    # Backpropagate
    optimizer.zero_grad()
    nodes = [value_loss] + actions
    gradients = [torch.ones(1)] + [None for _ in actions] # No gradients for reinforced values
    autograd.backward(nodes, gradients)
    optimizer.step()
    
    return discounted_rewards, value_loss

hidden_size = 50
learning_rate = 1e-4
weight_decay = 1e-5

log_every = 1000
render_every = 20000

env = Environment()
policy = Policy(hidden_size=hidden_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)#, weight_decay=weight_decay)

reward_avg = SlidingAverage('reward avg', steps=log_every)
value_avg = SlidingAverage('value avg', steps=log_every)

e = 0

while reward_avg < 0.75:
    actions, values, rewards = run_episode(e)
    final_reward = rewards[-1]
    
    discounted_rewards, value_loss = finish_episode(e, actions, values, rewards)
    
    reward_avg.add(final_reward)
    value_avg.add(value_loss.data[0])
    
    if e % log_every == 0:
        print('[epoch=%d]' % e, reward_avg, value_avg)
    
    if e > 0 and e % render_every == 0:
        animate(env.history)
    
    e += 1

# Plot average reward and value loss
plt.plot(np.array(reward_avg.avgs))
plt.show()
plt.plot(np.array(value_avg.avgs))
plt.show()