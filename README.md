# A DRL algorithm with Wolpertinger architecture in handling edge caching

## File Description:
- main.py - the main function to run the program
- wolpertinger.py - implements the wolpertinger architecture workflow
- ddpg.py - implements deep deterministic policy gradient algorithm
- cache_env.py - implements an environment for DRL agent to interact with
- memory.py - implements as a replay buffer(brain) in DDPG
- knn.py - implements the K-nearest neighbor component
- actor_network.py - implements the actor component
- critic_network.py - implements the critic component
- OU.py - implements the noise function
- generate_data.py - To generate request data in different request pattern
- plotGraph.py - plot the cache hit rate result to graph
- hypothesis_testing.py - Evaluate the confidence level of the results
- pythonShell.py - not related, a Python script for automation

- data/*.txt - They are generated request data in different patterns, the pattern file names ending with 2 are for online testing, and others are for offline training.

- offline_model/*.h5 - They are the trained networks' parameters in different request patterns. Including the original framework's trained networks and the proposed method's trained networks

- result/
  - *.hitrate.txt: They are the cache hit rates of both approaches after the online phase, used for hypothesis testing
  - *.png: They are the graph plot by plotGraph.py showing the online performance of both approaches
  - *_offline_final_hitrate.txt: They are the cache hit rates at the end of the offline training in both approaches and different request patterns
  - other txt files: They are the cache hit rates achieved by different hyperparameters.

## Usage:
- main.py<br>
run main.py by python3, input the cache size in integer, select the original framework or proposed method, pick one generated request dataset, run in offline training or online testing

		python3 main.py [cache size(int)] [original/proposed] [dataset] [offline/online]

  dataset:
  - zipf - request pattern follows a Zipf distribution
  - varNor - request pattern follows a variable normal distribution
  - mix - request pattern follows a Zipf distribution first and then changes to a normal distribution

- generate_data.py<br>
run by python3, select the pattern to generate, and input the file name to save the generated request

		python3 generate_data.py [pattern] [file name]

  - pattern: zipf / varNor / mix

- hypothesis_testing.py<br>
run by python3, select one pattern's result to do hypothesis testing

		python3 hypothesis_testing.py [pattern]

  - pattern: zipf / varNor / mix

- plotGraph.py<br>
run by python3 to plot the graph showing the performance of the two approaches in the online phase.

		python3 plotGraph.py

## Example:
Since the repository already has generated data and trained models<br>
To use the existing models, modify the models' file names by removing the prefix, e.g. zipf_actor.h5 -> actor.h5, zipf_critic_original.h5 -> critic_original.h5<br>
Then run the online testing of two approaches by:

		python3 main.py 150 proposed zipf online
		python3 main.py 150 original zipf online

show the results on the graph by:

		python3 plotGraph.py

If run it from generating data, for a Zipf request pattern:

		python3 generate_data.py zipf data/request_data_zipf
		python3 main.py 150 original zipf offline
		python3 main.py 150 proposed zipf offline
		python3 main.py 150 original zipf online
		python3 main.py 150 proposed zipf online
		python3 plotGraph.py
