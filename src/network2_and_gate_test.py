import numpy as np
import network2


training_data = np.array([
        [[[0.0], [0.0]], [[1.0], [0.0]]],
        [[[0.0], [1.0]], [[1.0], [0.0]]],
        [[[1.0], [0.0]], [[1.0], [0.0]]],
        [[[1.0], [1.0]], [[0.0], [1.0]]]
    ])

evaluation_data = [
        [[[0.0], [0.0]], 0],
        [[[0.0], [1.0]], 0],
        [[[1.0], [0.0]], 0],
        [[[1.0], [1.0]], 1]
    ]

net = network2.Network([2, 2, 2])
net.SGD(training_data, epochs=150, mini_batch_size=4, eta=0.8,
        lmbda=1,
        evaluation_data=evaluation_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)