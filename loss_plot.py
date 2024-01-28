import matplotlib.pyplot as plt

# Read data from the file
with open('loss_iteration_data.txt', 'r') as file:
    data = file.readlines()

# Separate iteration and loss
iterations, losses = zip(*[map(float, line.strip().split('\t')) for line in data])

# Plot the data
plt.plot(iterations, losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss-Iteration Graph')
plt.legend()
plt.savefig('loss_iteration_graph.png')
plt.show()